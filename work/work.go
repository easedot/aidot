// Example provided with help from Jason Waldrip.
// Package work manages a pool of goroutines to perform work.
package work

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
)

// Worker must be implemented by types that want to use
// the work pool.
type Worker interface {
	Task()
}

// Pool provides a pool of goroutines that can execute any Worker
// tasks that are submitted.
type Pool struct {
	work chan Worker
	wg   sync.WaitGroup
}

// New creates a new work pool.
func New(maxGoroutines int) *Pool {
	p := Pool{
		work: make(chan Worker, 10),
	}

	p.wg.Add(maxGoroutines)
	for i := 0; i < maxGoroutines; i++ {
		go func() {
			for w := range p.work {
				//log.Println("work exec")
				w.Task()
				//break until work chan close
			}
			log.Println("work part Done")
			p.wg.Done()
		}()
	}
	return &p
}

// Run submits work to the pool.
func (p *Pool) Run(w Worker) {
	p.work <- w
}

// Shutdown waits for all the goroutines to shutdown.
func (p *Pool) Shutdown() {
	close(p.work)
	p.wg.Wait()
}

func (p *Pool) WaitJob() {
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)
	select {
	// Signaled when an interrupt event is sent.
	case <-interrupt:
		// Stop receiving any further signals.
		fmt.Print("ctr break...")
		p.Shutdown()
		// Continue running as normal.
	}
}
