package main

import (
	ng "test_ai/numgo"
	nt "test_ai/tensor"
)

func main() {
	x := nt.NewOnes(1, 5)
	x.Print("x")
	y := ng.Dropout(x, 0.5)
	y.Print("train_y")
	ng.Backprop = false
	y = ng.Dropout(x, 0.5)
	y.Print("test_y")

}
