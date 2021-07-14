package main

import "C"
import (
	"fmt"
	"image/color"
	"math/rand"

	"gonum.org/v1/gonum/mat"
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

	I,H,O := 1,10,1
	var ScaleWFunc=func(_,_ int,v float64) float64{return v*0.01}

	W1 := ng.NewRandN(I, H)
	W1.Data.Apply(ScaleWFunc,W1.Data)
	W1.Name="W1"

	b1:=ng.NewZeros(1,H)
	b1.Name="b1"

	W2 := ng.NewRandN(H, O)
	W2.Data.Apply(ScaleWFunc,W2.Data)
	W2.Name="W2"

	b2:=ng.NewZeros(1,O)
	b2.Name="b2"

	predict:=func(x *ng.Variable)*ng.Variable{
		y:=ng.Linear(x,W1,b1)
		y=ng.Sigmoid(y)
		y=ng.Linear(y,W2,b2)
		return y
	}


	lr:=0.2
	var MulLrFunc=func(_,_ int,v float64) float64{return v*lr}

	iters:=10000
	for i:=0;i<iters;i++{
		yPred :=predict(x)
		loss:=ng.MeanSquaredError(y, yPred)
		if i==0{
			yPred.Plot(true,"./temp/pred.png")
			loss.Plot(true,"./temp/loss.png")
		}

		W1.ClearGrade()
		b1.ClearGrade()
		W2.ClearGrade()
		b2.ClearGrade()

		//设为false可以大幅加快速度
		loss.Backward(false)

		//更新参数不使用连接图，直接修改data,否则会造成计算图扩张，末端部署x,w,b等参数，
		//则backward时就不能设置false，因为如果设置了，他们友不是末级节点，梯度就会被清楚
		w1d:=mat.Dense{}
		w1d.Apply(MulLrFunc,W1.Grad.Data)
		W1.Data.Sub(W1.Data,&w1d)

		b1d:=mat.Dense{}
		b1d.Apply(MulLrFunc,b1.Grad.Data)
		b1.Data.Sub(b1.Data,&b1d)

		w2d:=mat.Dense{}
		w2d.Apply(MulLrFunc,W2.Grad.Data)
		W2.Data.Sub(W2.Data,&w2d)

		b2d:=mat.Dense{}
		b2d.Apply(MulLrFunc,b2.Grad.Data)
		b2.Data.Sub(b2.Data,&b2d)

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
		predPts[i].Y=predict(ng.NewVar(px)).At(0,0)
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

