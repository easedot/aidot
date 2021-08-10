package main

import (
	nd "test_ai/numed"
	ng "test_ai/numgo"
)

func main() {
	x := nd.NewOnes(1, 5)
	x.Print("x")
	y := ng.Dropout(x, 0.5)
	y.Print("train_y")
	ng.Backprop = false
	y = ng.Dropout(x, 0.5)
	y.Print("test_y")

}
