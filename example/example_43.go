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
	a, _ := ep.Parse("sin(2*3.1415926*x)+noise")
	x := ng.AsVar(nt.NewRand(100, 1))
	x.Name = "X"
	noise := ng.AsVar(nt.NewRand(100, 1))
	//y:=ng.Add(ng.Add(ng.NewVar(5),ng.Mul(ng.NewVar(2),x)),x1)
	y := a.Eval(ep.V{"x": x, "noise": noise})
	y = &ng.Variable{Data: y.Data}
	y.Name = "Y"

	I, H, O := 1, 10, 1

	W1 := ng.AsVar(nt.NewRandNorm(I, H))
	W1.Data = nt.Mul(W1.Data, nt.NewVar(0.01))
	W1.Name = "W1"

	b1 := ng.AsVar(nt.NewZeros(1, H))
	b1.Name = "b1"

	W2 := ng.AsVar(nt.NewRandNorm(H, O))
	//这里mul中的w2曾经写成了w1，造成了错误，找了半天，发现行列不对，10，1变成了1，10
	W2.Data = nt.Mul(W2.Data, nt.NewVar(0.01))
	W2.Name = "W2"

	b2 := ng.AsVar(nt.NewZeros(1, O))
	b2.Name = "b2"

	predict := func(x *ng.Variable) *ng.Variable {
		t := ng.Linear(x, W1, b1)
		t = ng.Sigmoid(t)
		t = ng.Linear(t, W2, b2)
		return t
	}

	lr := nt.NewVar(0.2)
	iters := 10000
	for i := 0; i < iters; i++ {
		yPred := predict(x)
		loss := ng.MeanSquaredError(y, yPred)
		if i == 0 {
			yPred.Plot(true, "./temp/pred.png")
			loss.Plot(true, "./temp/loss.png")
		}

		W1.ClearGrade()
		b1.ClearGrade()
		W2.ClearGrade()
		b2.ClearGrade()
		//设为false可以大幅加快速度
		loss.Backward(false)
		//更新参数不使用连接图，直接修改data,否则会造成计算图扩张，末端部署x,w,b等参数，
		//则backward时就不能设置false，因为如果设置了，他们友不是末级节点，梯度就会被清楚
		//更新参数不使用连接图，直接修改data
		W1.Data = nt.Sub(W1.Data, nt.Mul(W1.Grad.Data, lr))
		b1.Data = nt.Sub(b1.Data, nt.Mul(b1.Grad.Data, lr))
		W2.Data = nt.Sub(W2.Data, nt.Mul(W2.Grad.Data, lr))
		b2.Data = nt.Sub(b2.Data, nt.Mul(b2.Grad.Data, lr))
		if i%100 == 0 {
			fmt.Println(i, loss.Sprint("loss"))
		}
	}

	p := plot.New()
	p.Title.Text = "Plotutil example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	preY := predict(x)
	predPts := make(plotter.XYs, x.Data.Shape()[0])
	pts := make(plotter.XYs, x.Data.Shape()[0])
	for i := 0; i < x.Data.Shape()[0]; i++ {
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
