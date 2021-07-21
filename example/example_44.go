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
)

func main() {
	a,_:=ep.Parse("sin(2*3.1415926*x)+x1")
	x1:=ng.NewRand(100,1)
	x2:=ng.NewRand(100,1)
	y:=a.Eval(ep.V{"x":x1,"x1":x2})

	//切断计算图，从头开始
	y,x:=ng.CopyData(y),ng.CopyData(x1)
	x.Name,y.Name="X","Y"
	adm:=ng.Adam(0.01,0.9,0.999,1e-8)
	//sgd := ng.SGD(0.2)
	model:=ng.MLP(ng.Sigmoid, adm,10,1)
	model.Plot(x,true,"./temp/model.png")

	iters:=10000
	for i:=0;i<iters;i++{
		yPred:=model.Forward(x)
		loss:=ng.MeanSquaredError(y, yPred)

		model.ClearGrad()

		//设为false可以大幅加快速度
		loss.Backward(false)

		model.Grad2Param()

		if i%100==0{
			fmt.Println(i,loss.Sprint("loss"))
		}
	}

	p := plot.New()
	p.Title.Text = "Plotutil example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	predPts:=make(plotter.XYs,x.Shape().R)
	pts:=make(plotter.XYs,x.Shape().R)
	for i:=0;i<x.Shape().R;i++{
		px := x.At(i, 0)
		pts[i].X,predPts[i].X= px,px
		pts[i].Y=y.At(i,0)
		predPts[i].Y=model.Forward(ng.NewVar(px)).At(0,0)
	}
	ptsList:=[]plotter.XYs{pts,predPts}

	// Make a scatter plotter and set its style.
	var pls []plot.Plotter
	for _,sd:=range ptsList{
		s, err := plotter.NewScatter(sd)
		if err != nil {
			panic(err)
		}
		s.GlyphStyle.Color = color.RGBA{R: uint8(rand.Intn(255)), B: uint8(rand.Intn(255)), A: uint8(rand.Intn(255))}
		pls=append(pls,s)
	}
	p.Add(pls...)
	if err := p.Save(10*vg.Inch, 6*vg.Inch, "./temp/points.png"); err != nil {
		panic(err)
	}
}

