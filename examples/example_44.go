package main

import "C"
import (
	"fmt"
	"image/color"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	ep "test_ai/eval"
	ng "test_ai/numgo"
	nt "test_ai/tensor"
)

func main() {
	//todo 产生数据重构为tensor
	a, _ := ep.Parse("sin(2*3.1415926*x)+x1")
	x1 := ng.AsVar(nt.NewRand(100, 1))
	x2 := ng.AsVar(nt.NewRand(100, 1))
	y := a.Eval(ep.V{"x": x1, "x1": x2})

	//切断计算图，从头开始
	y, x := ng.AsVar(y.Data), ng.AsVar(x1.Data)
	x.Name, y.Name = "X", "Y"
	//adm:=ng.Adam(0.01,0.9,0.999,1e-8)
	sgd := ng.SGD(0.2)
	model := ng.MLP(ng.Sigmoid, sgd, 10, 1)
	model.Plot(x, true, "./temp/model.png")

	iters := 10000
	for i := 0; i < iters; i++ {
		yPred := model.Forward(x)
		loss := ng.MeanSquaredError(y, yPred)

		model.ClearGrad()

		//设为false可以大幅加快速度
		loss.Backward(false)

		model.Grad2Param()

		if i%1000 == 0 {
			fmt.Printf("idx:%d loss:%3.5f\n", i, loss.Data.Var())
		}
	}

	p := plot.New()
	p.Title.Text = "Plotutil examples"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	preY := model.Forward(x)
	predPts := make(plotter.XYs, x.Shape()[0])
	pts := make(plotter.XYs, x.Shape()[0])
	for i := 0; i < x.Shape()[0]; i++ {
		px := x.Data.Get(i, 0)
		pts[i].X, predPts[i].X = px, px
		pts[i].Y = y.Data.Get(i, 0)
		predPts[i].Y = preY.Data.Get(i, 0)
	}
	ptsList := []plotter.XYs{pts, predPts}

	// Make a scatter plotter and set its style.
	var pls []plot.Plotter
	for _, sd := range ptsList {
		s, err := plotter.NewScatter(sd)
		if err != nil {
			panic(err)
		}
		s.GlyphStyle.Color = color.RGBA{R: uint8(rand.Intn(255)), B: uint8(rand.Intn(255)), A: uint8(rand.Intn(255))}
		pls = append(pls, s)
	}
	p.Add(pls...)
	if err := p.Save(10*vg.Inch, 6*vg.Inch, "./temp/points.png"); err != nil {
		panic(err)
	}
}
