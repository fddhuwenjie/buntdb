package buntdb

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// onlineTestHooks shrinks the backfill chunk size and installs a deterministic
// per-chunk hook for the duration of a test.
func onlineTestHooks(t *testing.T, chunkItems int, hook func()) {
	t.Helper()
	oldItems := onlineIndexChunkItems
	oldHook := onlineIndexChunkHook
	onlineIndexChunkItems = chunkItems
	onlineIndexChunkHook = hook
	t.Cleanup(func() {
		onlineIndexChunkItems = oldItems
		onlineIndexChunkHook = oldHook
	})
}

type onlineKV struct{ k, v string }

// onlineSnapshot returns every live (non-expired) key/value pair in the
// database, it is the source of truth for index verification.
func onlineSnapshot(t *testing.T, db *DB) []onlineKV {
	t.Helper()
	var pairs []onlineKV
	if err := db.View(func(tx *Tx) error {
		return tx.Ascend("", func(key, value string) bool {
			pairs = append(pairs, onlineKV{key, value})
			return true
		})
	}); err != nil {
		t.Fatal(err)
	}
	return pairs
}

func onlineSnapshotPrefix(t *testing.T, db *DB, prefix string) []onlineKV {
	pairs := onlineSnapshot(t, db)
	out := pairs[:0]
	for _, p := range pairs {
		if len(p.k) >= len(prefix) && p.k[:len(prefix)] == prefix {
			out = append(out, p)
		}
	}
	return out
}

// onlineAssertBTree verifies that the regular index contains exactly the
// matching snapshot entries and that they are ordered by the index less
// function with a key tie-breaker.
func onlineAssertBTree(t *testing.T, db *DB, name, prefix string,
	less func(a, b string) bool) {
	t.Helper()
	want := onlineSnapshotPrefix(t, db, prefix)
	sort.Slice(want, func(i, j int) bool {
		if less(want[i].v, want[j].v) {
			return true
		}
		if less(want[j].v, want[i].v) {
			return false
		}
		return want[i].k < want[j].k
	})
	var got []onlineKV
	if err := db.View(func(tx *Tx) error {
		return tx.Ascend(name, func(key, value string) bool {
			got = append(got, onlineKV{key, value})
			return true
		})
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("index %q: expected %d items, got %d", name, len(want), len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %q: position %d expected %q=%q, got %q=%q",
				name, i, want[i].k, want[i].v, got[i].k, got[i].v)
		}
	}
}

func onlineRectsOverlap(min1, max1, min2, max2 []float64) bool {
	for d := range min1 {
		if max1[d] < min2[d] || min1[d] > max2[d] {
			return false
		}
	}
	return true
}

// onlineAssertRTree verifies that the spatial index contains exactly the
// matching snapshot entries for a world-wide query, and that a window query
// matches precisely the expected rectangles.
func onlineAssertRTree(t *testing.T, db *DB, name, prefix string) {
	t.Helper()
	want := onlineSnapshotPrefix(t, db, prefix)
	wantSet := make(map[string]string, len(want))
	for _, p := range want {
		wantSet[p.k] = p.v
	}
	world := Rect([]float64{-1e12, -1e12}, []float64{1e12, 1e12})
	gotSet := make(map[string]string)
	if err := db.View(func(tx *Tx) error {
		return tx.Intersects(name, world, func(key, value string) bool {
			gotSet[key] = value
			return true
		})
	}); err != nil {
		t.Fatal(err)
	}
	if len(gotSet) != len(wantSet) {
		t.Fatalf("spatial index %q: expected %d items, got %d",
			name, len(wantSet), len(gotSet))
	}
	for k, v := range wantSet {
		if gv, ok := gotSet[k]; !ok || gv != v {
			t.Fatalf("spatial index %q: missing or wrong value for %q", name, k)
		}
	}
	// Window query: only rectangles intersecting [35,35]-[60,60].
	qmin, qmax := []float64{35, 35}, []float64{60, 60}
	bounds := Rect(qmin, qmax)
	var windowKeys []string
	if err := db.View(func(tx *Tx) error {
		return tx.Intersects(name, bounds, func(key, value string) bool {
			windowKeys = append(windowKeys, key)
			return true
		})
	}); err != nil {
		t.Fatal(err)
	}
	var expectWindow []string
	for _, p := range want {
		min, max := IndexRect(p.v)
		if onlineRectsOverlap(min, max, qmin, qmax) {
			expectWindow = append(expectWindow, p.k)
		}
	}
	sort.Strings(windowKeys)
	sort.Strings(expectWindow)
	if len(windowKeys) != len(expectWindow) {
		t.Fatalf("spatial index %q window: expected %d keys, got %d",
			name, len(expectWindow), len(windowKeys))
	}
	for i := range expectWindow {
		if windowKeys[i] != expectWindow[i] {
			t.Fatalf("spatial index %q window mismatch at %d: %q vs %q",
				name, i, windowKeys[i], expectWindow[i])
		}
	}
}

func onlineAssertNoBuilds(t *testing.T, db *DB) {
	t.Helper()
	db.mu.Lock()
	n := len(db.builds)
	db.mu.Unlock()
	if n != 0 {
		t.Fatalf("expected no active online builds, found %d", n)
	}
}

func onlineOpen(t *testing.T, memory bool) (*DB, func()) {
	t.Helper()
	if memory {
		db, err := Open(":memory:")
		if err != nil {
			t.Fatal(err)
		}
		return db, func() { _ = db.Close() }
	}
	db := testOpen(t)
	return db, func() { testClose(db) }
}

func onlineSeed(t *testing.T, db *DB, n int) {
	t.Helper()
	if err := db.Update(func(tx *Tx) error {
		for i := 0; i < n; i++ {
			if _, _, err := tx.Set(
				fmt.Sprintf("key:%04d", i), strconv.Itoa(i), nil); err != nil {
				return err
			}
			if _, _, err := tx.Set(
				fmt.Sprintf("geo:%04d", i),
				Point(float64(i), float64(i)), nil); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// TestReplaceIndexOnlineTransitions drives every kind of committed mutation
// (add, overwrite, delete, DeleteAll and expiry cleanup) plus a rolled-back
// write through deterministic chunk-boundary hooks while both online
// replacements are building. The published indexes must exactly match the
// live database and its ordering / rectangle semantics.
func TestReplaceIndexOnlineTransitions(t *testing.T) {
	for _, memory := range []bool{true, false} {
		name := "disk"
		if memory {
			name = "memory"
		}
		t.Run(name, func(t *testing.T) {
			db, closeDB := onlineOpen(t, memory)
			defer closeDB()
			onlineSeed(t, db, 200)
			// the previous generation uses string ordering / a plain index.
			if err := db.CreateIndex("idx", "key:*", IndexString); err != nil {
				t.Fatal(err)
			}
			if err := db.CreateSpatialIndex("geo", "geo:*", IndexRect); err != nil {
				t.Fatal(err)
			}

			// Two online builds share this hook, so script steps must run
			// exactly once even though both goroutines fire the hook.
			var hookMu sync.Mutex
			tick := 0
			scriptDone := make(map[int]bool)
			runStep := func(n int, fn func()) {
				hookMu.Lock()
				defer hookMu.Unlock()
				if scriptDone[n] {
					return
				}
				scriptDone[n] = true
				fn()
			}
			hook := func() {
				// The previous index and ordinary reads must remain
				// available and consistent for the whole build.
				if err := db.View(func(tx *Tx) error {
					if err := tx.Ascend("idx", func(k, v string) bool {
						return true
					}); err != nil {
						return err
					}
					return tx.Intersects("geo",
						Rect([]float64{-1e9, -1e9}, []float64{1e9, 1e9}),
						func(k, v string) bool { return true })
				}); err != nil {
					t.Errorf("index unavailable during build: %v", err)
				}
				hookMu.Lock()
				tick++
				n := tick
				hookMu.Unlock()
				switch n {
				case 1:
					// committed add, overwrite and delete
					runStep(1, func() {
						if err := db.Update(func(tx *Tx) error {
							if _, _, err := tx.Set("key:0900", "900", nil); err != nil {
								return err
							}
							if _, _, err := tx.Set("geo:0900",
								Point(900, 900), nil); err != nil {
								return err
							}
							if _, _, err := tx.Set("key:0000", "9999", nil); err != nil {
								return err
							}
							if _, _, err := tx.Set("geo:0000",
								Point(99, 99), nil); err != nil {
								return err
							}
							if _, err := tx.Delete("key:0001"); err != nil {
								return err
							}
							if _, err := tx.Delete("geo:0001"); err != nil {
								return err
							}
							return nil
						}); err != nil {
							t.Errorf("step1: %v", err)
						}
					})
				case 2:
					// rolled back write: must never reach the new indexes
					runStep(2, func() {
						wtx, err := db.Begin(true)
						if err != nil {
							t.Errorf("step2 begin: %v", err)
							return
						}
						if _, _, err := wtx.Set("key:rollback", "1", nil); err != nil {
							t.Errorf("step2 set: %v", err)
							_ = wtx.Rollback()
							return
						}
						if _, _, err := wtx.Set("geo:rollback",
							Point(1, 1), nil); err != nil {
							t.Errorf("step2 set geo: %v", err)
							_ = wtx.Rollback()
							return
						}
						if _, err := wtx.Delete("key:0002"); err != nil {
							t.Errorf("step2 del: %v", err)
							_ = wtx.Rollback()
							return
						}
						if err := wtx.Rollback(); err != nil {
							t.Errorf("step2 rollback: %v", err)
							return
						}
						// the rollback must be fully invisible immediately
						if err := db.View(func(tx *Tx) error {
							if _, err := tx.Get("key:rollback"); err != ErrNotFound {
								return errors.New("rolled back set leaked")
							}
							if _, err := tx.Get("geo:rollback"); err != ErrNotFound {
								return errors.New("rolled back geo set leaked")
							}
							if v, err := tx.Get("key:0002"); err != nil || v != "2" {
								return errors.New("rolled back delete leaked")
							}
							return nil
						}); err != nil {
							t.Errorf("step2 verify: %v", err)
						}
					})
				case 3:
					// whole database clear
					runStep(3, func() {
						if err := db.Update(func(tx *Tx) error {
							return tx.DeleteAll()
						}); err != nil {
							t.Errorf("step3: %v", err)
						}
					})
				case 4:
					// repopulate after the clear
					runStep(4, func() {
						if err := db.Update(func(tx *Tx) error {
							for i := 1000; i < 1300; i++ {
								if _, _, err := tx.Set(
									fmt.Sprintf("key:%04d", i),
									strconv.Itoa(i), nil); err != nil {
									return err
								}
								if _, _, err := tx.Set(
									fmt.Sprintf("geo:%04d", i),
									Point(float64(i), float64(i)), nil); err != nil {
									return err
								}
							}
							// a few items inside the window query range
							for i := 40; i < 46; i++ {
								if _, _, err := tx.Set(
									fmt.Sprintf("key:%04d", i),
									strconv.Itoa(i), nil); err != nil {
									return err
								}
								if _, _, err := tx.Set(
									fmt.Sprintf("geo:%04d", i),
									Point(float64(i), float64(i)), nil); err != nil {
									return err
								}
							}
							return nil
						}); err != nil {
							t.Errorf("step4: %v", err)
						}
					})
				case 5:
					// an expiring item; sleep long enough for the
					// background manager to commit its deletion while the
					// build is still scanning.
					runStep(5, func() {
						if err := db.Update(func(tx *Tx) error {
							_, _, err := tx.Set("geo:expx", Point(42, 42),
								&SetOptions{Expires: true, TTL: 100 * time.Millisecond})
							return err
						}); err != nil {
							t.Errorf("step5 set: %v", err)
							return
						}
						time.Sleep(1250 * time.Millisecond)
						var gone bool
						_ = db.View(func(tx *Tx) error {
							_, err := tx.Get("geo:expx")
							gone = err == ErrNotFound
							return nil
						})
						if !gone {
							t.Error("expired item was not cleaned up during build")
						}
					})
				}
			}
			onlineTestHooks(t, 8, hook)

			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				defer wg.Done()
				if err := db.ReplaceIndexOnline(context.Background(),
					"idx", "key:*", IndexInt); err != nil {
					t.Errorf("ReplaceIndexOnline: %v", err)
				}
			}()
			go func() {
				defer wg.Done()
				if err := db.ReplaceSpatialIndexOnline(context.Background(),
					"geo", "geo:*", IndexRect); err != nil {
					t.Errorf("ReplaceSpatialIndexOnline: %v", err)
				}
			}()
			wg.Wait()

			// the finished indexes must exactly match the live state.
			onlineAssertBTree(t, db, "idx", "key:", IndexInt)
			onlineAssertRTree(t, db, "geo", "geo:")

			// the rolled back writes must not exist anywhere. (The rolled
			// back delete is verified immediately after the rollback; the
			// key is legitimately removed by the later DeleteAll.)
			if err := db.View(func(tx *Tx) error {
				for _, k := range []string{"key:rollback", "geo:rollback"} {
					if _, err := tx.Get(k); err != ErrNotFound {
						return fmt.Errorf("%s should be missing", k)
					}
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			onlineAssertNoBuilds(t, db)
		})
	}
}

// TestReplaceIndexOnlineCreateMissing verifies the "absent -> create" path.
func TestReplaceIndexOnlineCreateMissing(t *testing.T) {
	db, closeDB := onlineOpen(t, true)
	defer closeDB()
	onlineSeed(t, db, 50)

	if err := db.ReplaceIndexOnline(context.Background(),
		"idx2", "key:*", IndexInt); err != nil {
		t.Fatal(err)
	}
	if err := db.ReplaceSpatialIndexOnline(context.Background(),
		"geo2", "geo:*", IndexRect); err != nil {
		t.Fatal(err)
	}
	names, err := db.Indexes()
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(names)
	if fmt.Sprint(names) != "[geo2 idx2]" {
		t.Fatalf("unexpected indexes: %v", names)
	}
	onlineAssertBTree(t, db, "idx2", "key:", IndexInt)
	onlineAssertRTree(t, db, "geo2", "geo:")
}

// TestReplaceIndexOnlineConcurrent mixes several random writers, readers and
// the online builds on both in-memory and disk databases. After the builds
// publish, every index must equal the live state, ordering must be correct,
// and subsequent writes must keep maintaining the published indexes.
func TestReplaceIndexOnlineConcurrent(t *testing.T) {
	for _, memory := range []bool{true, false} {
		name := "disk"
		if memory {
			name = "memory"
		}
		t.Run(name, func(t *testing.T) {
			for iter := 0; iter < 2; iter++ {
				db, closeDB := onlineOpen(t, memory)
				onlineSeed(t, db, 300)
				if err := db.CreateIndex("idx", "key:*", IndexString); err != nil {
					t.Fatal(err)
				}
				if err := db.CreateSpatialIndex("geo", "geo:*", IndexRect); err != nil {
					t.Fatal(err)
				}
				onlineTestHooks(t, 64, nil)

				stop := make(chan struct{})
				var wg sync.WaitGroup
				for w := 0; w < 4; w++ {
					wg.Add(1)
					go func(seed int64) {
						defer wg.Done()
						rng := rand.New(rand.NewSource(seed))
						for {
							select {
							case <-stop:
								return
							default:
							}
							i := rng.Intn(500)
							_ = db.Update(func(tx *Tx) error {
								switch rng.Intn(10) {
								case 0:
									_, _ = tx.Delete(fmt.Sprintf("key:%04d", i))
									_, _ = tx.Delete(fmt.Sprintf("geo:%04d", i))
								case 1:
									if i == 0 {
										return tx.DeleteAll()
									}
									_, _, _ = tx.Set(fmt.Sprintf("key:%04d", i),
										strconv.Itoa(rng.Intn(1000)), nil)
									_, _, _ = tx.Set(fmt.Sprintf("geo:%04d", i),
										Point(rng.Float64()*100, rng.Float64()*100),
										nil)
								default:
									_, _, _ = tx.Set(fmt.Sprintf("key:%04d", i),
										strconv.Itoa(rng.Intn(1000)), nil)
									_, _, _ = tx.Set(fmt.Sprintf("geo:%04d", i),
										Point(float64(i), float64(i)), nil)
								}
								return nil
							})
						}
					}(int64(iter*100 + w))
				}
				for r := 0; r < 2; r++ {
					wg.Add(1)
					go func() {
						defer wg.Done()
						for {
							select {
							case <-stop:
								return
							default:
							}
							_ = db.View(func(tx *Tx) error {
								_ = tx.Ascend("idx", func(k, v string) bool {
									return true
								})
								return tx.Intersects("geo",
									Rect([]float64{-1e9, -1e9}, []float64{1e9, 1e9}),
									func(k, v string) bool { return true })
							})
						}
					}()
				}

				if err := db.ReplaceIndexOnline(context.Background(),
					"idx", "key:*", IndexInt); err != nil {
					close(stop)
					wg.Wait()
					t.Fatalf("ReplaceIndexOnline: %v", err)
				}
				if err := db.ReplaceSpatialIndexOnline(context.Background(),
					"geo", "geo:*", IndexRect); err != nil {
					close(stop)
					wg.Wait()
					t.Fatalf("ReplaceSpatialIndexOnline: %v", err)
				}
				close(stop)
				wg.Wait()

				onlineAssertBTree(t, db, "idx", "key:", IndexInt)
				onlineAssertRTree(t, db, "geo", "geo:")

				// the published indexes must keep following live writes.
				if err := db.Update(func(tx *Tx) error {
					_, _, err := tx.Set("key:0401", "401", nil)
					if err != nil {
						return err
					}
					_, _, err = tx.Set("geo:0401", Point(401, 401), nil)
					return err
				}); err != nil {
					t.Fatal(err)
				}
				onlineAssertBTree(t, db, "idx", "key:", IndexInt)
				onlineAssertRTree(t, db, "geo", "geo:")
				onlineAssertNoBuilds(t, db)
				closeDB()
			}
		})
	}
}

// TestReplaceIndexOnlineCancel ensures cancellation aborts safely, publishes
// nothing and leaves the previous index fully usable.
func TestReplaceIndexOnlineCancel(t *testing.T) {
	db, closeDB := onlineOpen(t, true)
	defer closeDB()
	onlineSeed(t, db, 400)
	if err := db.CreateIndex("idx", "key:*", IndexString); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	tick := 0
	onlineTestHooks(t, 4, func() {
		tick++
		if tick == 2 {
			cancel()
		}
	})
	if err := db.ReplaceIndexOnline(ctx, "idx", "key:*", IndexInt); err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	// old definition and data must still be there.
	onlineAssertBTree(t, db, "idx", "key:", IndexString)
	onlineAssertNoBuilds(t, db)
	// the synchronous entry must still work afterwards.
	if err := db.ReplaceIndex("idx", "key:*", IndexInt); err != nil {
		t.Fatal(err)
	}
	onlineAssertBTree(t, db, "idx", "key:", IndexInt)
}

// TestReplaceIndexOnlineClose ensures closing the database aborts safely.
func TestReplaceIndexOnlineClose(t *testing.T) {
	db, _ := onlineOpen(t, true)
	onlineSeed(t, db, 400)
	if err := db.CreateIndex("idx", "key:*", IndexString); err != nil {
		t.Fatal(err)
	}
	tick := 0
	onlineTestHooks(t, 4, func() {
		tick++
		if tick == 2 {
			_ = db.Close()
		}
	})
	if err := db.ReplaceIndexOnline(context.Background(),
		"idx", "key:*", IndexInt); err != ErrDatabaseClosed {
		t.Fatalf("expected ErrDatabaseClosed, got %v", err)
	}
}

// TestReplaceIndexOnlineConflict ensures a same-named index changed by
// another operation is never overwritten by the unfinished build.
func TestReplaceIndexOnlineConflict(t *testing.T) {
	db, closeDB := onlineOpen(t, true)
	defer closeDB()
	onlineSeed(t, db, 300)
	if err := db.CreateIndex("idx", "key:*", IndexString); err != nil {
		t.Fatal(err)
	}

	tick := 0
	onlineTestHooks(t, 8, func() {
		tick++
		if tick == 2 {
			// another operation drops and recreates the name with a
			// different definition.
			if err := db.DropIndex("idx"); err != nil {
				t.Fatal(err)
			}
			if err := db.CreateIndex("idx", "other:*", IndexInt); err != nil {
				t.Fatal(err)
			}
		}
	})
	err := db.ReplaceIndexOnline(context.Background(), "idx", "key:*", IndexInt)
	if err != ErrIndexModified {
		t.Fatalf("expected ErrIndexModified, got %v", err)
	}
	// the newer definition must be intact.
	if err := db.Update(func(tx *Tx) error {
		_, _, err := tx.Set("other:1", "1", nil)
		if err != nil {
			return err
		}
		_, _, err = tx.Set("key:1", "1", nil)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := db.View(func(tx *Tx) error {
		return tx.Ascend("idx", func(k, v string) bool { n++; return true })
	}); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("newer index definition must be kept, got %d entries", n)
	}
	onlineAssertNoBuilds(t, db)

	// a plain drop of the name during the build must also abort.
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	tick = 0
	onlineTestHooks(t, 8, func() {
		tick++
		if tick == 2 {
			if err := db.DropIndex("idx"); err != nil {
				t.Fatal(err)
			}
		}
	})
	if err := db.ReplaceIndexOnline(ctx2, "idx", "key:*", IndexInt); err != ErrIndexModified {
		t.Fatalf("expected ErrIndexModified after drop, got %v", err)
	}
	if names, err := db.Indexes(); err != nil || len(names) != 0 {
		t.Fatalf("index should be gone, names=%v err=%v", names, err)
	}
	onlineAssertNoBuilds(t, db)
}

// TestReplaceIndexOnlineRace runs two online replacements of the same name
// concurrently. Exactly one must win; the loser must abort with
// ErrIndexModified and must not clobber the winner.
func TestReplaceIndexOnlineRace(t *testing.T) {
	db, closeDB := onlineOpen(t, true)
	defer closeDB()
	onlineSeed(t, db, 500)
	onlineTestHooks(t, 4, nil)
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = db.ReplaceIndexOnline(context.Background(),
				"idx", "key:*", IndexInt)
		}(i)
	}
	wg.Wait()
	nOK, nModified := 0, 0
	for _, err := range errs {
		switch err {
		case nil:
			nOK++
		case ErrIndexModified:
			nModified++
		default:
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if nOK != 1 || nModified != 1 {
		t.Fatalf("expected one success and one conflict, got ok=%d modified=%d",
			nOK, nModified)
	}
	onlineAssertBTree(t, db, "idx", "key:", IndexInt)
	onlineAssertNoBuilds(t, db)
}

// TestReplaceIndexOnlineLoadConflict ensures that a Load (which replaces the
// whole dataset and wipes indexes outside a transaction) aborts an in-progress
// online build instead of publishing against the reset state.
func TestReplaceIndexOnlineLoadConflict(t *testing.T) {
	db, closeDB := onlineOpen(t, true)
	defer closeDB()
	onlineSeed(t, db, 300)
	if err := db.CreateIndex("idx", "key:*", IndexString); err != nil {
		t.Fatal(err)
	}
	tick := 0
	onlineTestHooks(t, 8, func() {
		tick++
		if tick == 2 {
			payload := "*1\r\n$7\r\nflushdb\r\n" +
				"*3\r\n$3\r\nset\r\n$4\r\nload\r\n$1\r\n1\r\n"
			if err := db.Load(strings.NewReader(payload)); err != nil {
				t.Errorf("Load: %v", err)
			}
		}
	})
	err := db.ReplaceIndexOnline(context.Background(),
		"idx", "key:*", IndexInt)
	if err != ErrIndexModified {
		t.Fatalf("expected ErrIndexModified, got %v", err)
	}
	if names, err := db.Indexes(); err != nil || len(names) != 0 {
		t.Fatalf("load flushdb should wipe indexes, names=%v err=%v", names, err)
	}
	if err := db.View(func(tx *Tx) error {
		if n, err := tx.Len(); err != nil || n != 1 {
			return fmt.Errorf("expected 1 loaded item, n=%d err=%v", n, err)
		}
		if v, err := tx.Get("load"); err != nil || v != "1" {
			return fmt.Errorf("loaded value mismatch: %q err=%v", v, err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	onlineAssertNoBuilds(t, db)
}

// TestReplaceIndexOnlinePublishRollback verifies that rolling back the
// publishing transaction restores the previous index object without a
// rebuild and restores the generation counter.
func TestReplaceIndexOnlinePublishRollback(t *testing.T) {
	db, closeDB := onlineOpen(t, true)
	defer closeDB()
	onlineSeed(t, db, 50)
	if err := db.CreateIndex("idx", "key:*", IndexString); err != nil {
		t.Fatal(err)
	}
	// build a ready candidate under the write lock
	tx, err := db.Begin(true)
	if err != nil {
		t.Fatal(err)
	}
	cand, err := newDatabaseIndex(db, "idx", "key:*",
		[]func(a, b string) bool{IndexInt}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cand.rebuild()
	gen0 := tx.db.idxGens["idx"]
	tx.installBuiltIndex("idx", cand, gen0)
	onlineAssertBTreeTx(t, tx, "key:", IndexInt)
	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
	// previous index restored as-is, string ordering
	onlineAssertBTree(t, db, "idx", "key:", IndexString)
	db.mu.RLock()
	gen := db.idxGens["idx"]
	db.mu.RUnlock()
	if gen != gen0 {
		t.Fatalf("generation not restored: %d != %d", gen, gen0)
	}
	// rollback bookkeeping must be fresh on the next transaction.
	if err := db.Update(func(tx *Tx) error {
		if len(tx.wc.rollbackRestore) != 0 || len(tx.wc.rollbackGens) != 0 {
			return errors.New("rollback context leaked")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func onlineAssertBTreeTx(t *testing.T, tx *Tx, prefix string,
	less func(a, b string) bool) {
	t.Helper()
	var got []onlineKV
	if err := tx.Ascend("idx", func(k, v string) bool {
		got = append(got, onlineKV{k, v})
		return true
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("expected published index to be visible inside the tx")
	}
	// verify int ordering
	for i := 1; i < len(got); i++ {
		if less(got[i].v, got[i-1].v) {
			t.Fatalf("published index not ordered: %q before %q",
				got[i].v, got[i-1].v)
		}
	}
}
