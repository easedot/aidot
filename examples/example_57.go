package main

import (
	"fmt"
	ng "test_ai/numgo"
	nt "test_ai/tensor"
)

func main() {
	x1 := nt.NewRand(1, 3, 7, 7)
	xv := ng.NewVariable(x1)
	col1 := ng.Img2col(xv, 5, 5, 1, 0, true)
	fmt.Printf("col %v\n", col1.Shape())

	x2 := nt.NewRand(10, 3, 7, 7)
	xv2 := ng.NewVariable(x2)
	col2 := ng.Img2col(xv2, 5, 5, 1, 0, true)
	fmt.Printf("col %v\n", col2.Shape())

	N, C, H, W := 1, 5, 15, 15
	OC, KH, KW := 8, 3, 3
	//_, KH, KW := 8, 3, 3
	x := ng.NewVariable(nt.NewRand(N, C, H, W))
	Wi := ng.NewVariable(nt.NewRand(OC, C, KH, KW))
	y := ng.Conv2d([]*ng.Variable{x, Wi, nil}, 1, 1)
	//y := ng.Conv2d_sample(x, Wi, nil, 1, 1)
	//y := ng.Pooling_simple(x, KH, KW, 1, 0)
	y.Backward(true)
	fmt.Printf("y shape %v\n", y.Shape())
	fmt.Printf("x.Grade shape %v\n", x.Grad.Shape())
}
