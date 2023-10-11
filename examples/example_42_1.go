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
	a, _ := ep.Parse("sin(2*3.1415926*x)+x1")
	x := ng.AsVar(nt.NewRand(100, 1))
	x.Name = "x"
	x1 := ng.AsVar(nt.NewRand(100, 1))
	x1.Name = "x1"
	//y:=ng.Add(ng.Add(ng.NewVar(5),ng.Mul(ng.NewVar(2),x)),x1)
	ev := a.Eval(ep.V{"x": x, "x1": x1})
	y := ng.AsVar(ev.Data)
	y.Name = "y"
	//ut.PrintDense("x",g.Data)
	//ut.PrintDense("x",x.Data)
	//ut.PrintDense("y",y.Data)
	W := ng.AsVar(nt.NewZeros(1, 1))
	W.Name = "w"
	b := ng.AsVar(nt.NewZeros(1))
	b.Name = "b"
	predict := func(x *ng.Variable) *ng.Variable {
		t := ng.Add(ng.Matmul(x, W), b)
		return t
	}
	lr := ng.NewVar(0.1)
	iters := 100
	for i := 0; i < iters; i++ {
		yPred := predict(x)
		loss := ng.MeanSquaredError(y, yPred)
		if i == 0 {
			yPred.Plot(true, "./temp/pred.png")
			loss.Plot(true, "./temp/loss.png")
		}

		W.ClearGrade()
		b.ClearGrade()

		loss.Backward(false)

		//更新参数不使用连接图，直接修改data
		W.Data = nt.Sub(W.Data, nt.Mul(W.Grad.Data, lr.Data))
		b.Data = nt.Sub(b.Data, nt.Mul(b.Grad.Data, lr.Data))

		fmt.Println(i, loss.Sprint("loss"), W.Sprint("w"), b.Sprint("b"))
		if loss.Data.Var() < 0.1 {
			break
		}
	}

	p := plot.New()
	p.Title.Text = "Plotutil examples"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	preY := predict(x)
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
