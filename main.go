package main

import "C"
import (
	"gonum.org/v1/gonum/mat"
	//mat "github.com/nlpodyssey/spago/pkg/mat32"
	ng "test_ai/numgo"
	//F "test_ai/functions"
)




func main() {
	t:=ng.NewVar(1)
	t.Print("x")
	t.GrowTo(3,3)
	t.Print("tg")
	m := mat.NewDense(2, 3, []float64{
		1,2,3,
		3,2,1,
	})

	//m := mat.NewDense(1, 1, []float64{
	//	2.0,
	//})
	//d := mat.NewDense(1,1,[]float64{
	//	3.0,
	//})
	x := &ng.Variable{Data:m}
	//x.Data=x.Data.Grow(3,3).(*mat.Dense)
	//x.Data.Scale(2, x.Data)
	//x.Name="x"
	//print("x:",dotVar(x,true))
	//y1 := pow(x,2)
	//y1.Name="y"
	//print("y:",dotVar(y1,true))
	//print("func:",dotFunc(y1.Creator))
	//print(getDotGraph(y1,true))
	//plotDotGraph(y1,true,"/Users/haihui/Downloads/pow.png")
	////printDense("y",y.Data) //4

	//check A numdiff
	//A := NewFunction(&Pow{C:2})
	//y := A.Run(x)
	//y.Backward(true)
	//printDense("xg",x.Grad.Data)
	//f := A.Run
	//g := numericalDiff(f,x,)
	//printDense("dg",g)

	////check b numdiff
	//A := NewFunction(&Exp{})
	//y:=A.Run(x)
	//y.Backward(false)
	//printDense("xg", x.Grad.Data)
	//f := A.Run
	//g := numericalDiff(f,x)
	//printDense("dg",g)

	com:= func (x...*ng.Variable) *ng.Variable {
		a := mul(x[0],x[0])
		b := exp(a)
		y := mul(b,b)
		return y
	}
	y := com(x)

	dy := numericalDiff(com,x)
	printDense("dg",dy)

	y.Backward(true)
	gx:=x.Grad
	printDense("gx",gx.Data)

	x.ClearGrade()
	gx.Backward(true)
	gx2:=x.Grad
	printDense("gx2",gx2.Data)
	plotDotGraph(gx2,true,"/Users/haihui/Downloads/pow.png")


	////多出多入测试
	//a := pow(x,2)
	//y := add(pow(a,2),pow(a,2))
	//y.Backward(true)
	//printDense("add",y.Data)
	////printDense("x0g",y[0].Grad)
	//printDense("x1g",x.Grad)


	//a := pow(x,2)
	//b := exp(a)
	//y := pow(b,2)
	//printDense("y", y.Data)
	//y.Backward(true)
	//printDense("xg",x.Grad)


}

