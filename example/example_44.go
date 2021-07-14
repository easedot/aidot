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
	a,_:=ep.Parse("sin(2*3.1415926*x)+noise")
	x:=ng.NewRand(100,1)
	x.Name="X"
	noise:=ng.NewRand(100,1)
	//y:=ng.Add(ng.Add(ng.NewVar(5),ng.Mul(ng.NewVar(2),x)),x1)
	y:=a.Eval(ep.Env{"x":x,"noise":noise})
	y=&ng.Variable{Data:y.Data}
	y.Name="Y"

	model:=ng.MLP(nil,10,1)
	//y.Plot(true,"./model.png")

	iters:=10000
	for i:=0;i<iters;i++{
		yPred:=model.Forward(x)
		//yPred :=predict(x)
		loss:=ng.MeanSquaredError(y, yPred)

		if i==0{
			yPred.Plot(true,"./temp/pred.png")
			loss.Plot(true,"./temp/loss.png")
		}
		model.ClearGrad()

		//设为false可以大幅加快速度
		loss.Backward(false)

		model.Grad2Param()

		if i%100==0{
			fmt.Println(i,loss.Sprint("loss"))
		}
		if loss.At(0,0)<0.01{
			break
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

