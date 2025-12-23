package dataobjects

import (
	"runtime"
	"sync"
)

const USE_WORKER_THREAD = true

type task struct {
	fn   func() interface{}
	done chan interface{}
}

type Worker struct {
	tasks chan task
	stop  chan struct{}
}

var (
	singleton *Worker
	mu        sync.Mutex
)

// GetDoWorker returns the singleton, creating it on demand.
// If the previous worker was shut down, a new one is created.
func GetDoWorker() *Worker {
	mu.Lock()
	defer mu.Unlock()

	if singleton == nil {
		w := &Worker{
			tasks: make(chan task),
			stop:  make(chan struct{}),
		}
		singleton = w

		if USE_WORKER_THREAD {
			go func() {
				runtime.LockOSThread()
				for {
					select {
					case t := <-w.tasks:
						t.done <- t.fn()
					case <-w.stop:
						return
					}
				}
			}()
		}
	}
	return singleton
}

// Run executes a closure on the locked thread.
func (w *Worker) Run(fn func() interface{}) interface{} {
	if USE_WORKER_THREAD {
		done := make(chan interface{}, 1)
		w.tasks <- task{fn: fn, done: done}
		return <-done
	} else {
		return fn()
	}
}

// Shutdown stops the worker and clears the singleton.
// Next call to GetWorker() will relaunch a new worker.
func (w *Worker) Shutdown() {
	mu.Lock()
	defer mu.Unlock()
	if singleton != nil {
		if USE_WORKER_THREAD {
			close(w.stop)
		}
		singleton = nil
	}
}
