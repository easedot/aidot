package main

import "C"
import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	//mat "github.com/nlpodyssey/spago/pkg/mat32"

)

type str string

type Variable struct{
	Data *mat.Dense
	Grad *mat.Dense
	Creator *Function
}

func (v *Variable) Backward() {
	if v.Grad==nil{
		v.Grad=LikeOnes(v.Data)
	}
	f:=v.Creator
	for f!=nil {
		x,y:=f.input,f.ouput
		x.Grad=f.Back(y).Grad
		f=x.Creator
	}
}

//func (v *Variable) Backward() {
//	stack:=[]*Function{v.Creator}
//	for len(stack)>0 {
//		f:=stack[len(stack)-1]
//		x,y:=f.input,f.ouput
//		x.Grad=f.Back(y).Grad
//		stack=stack[:len(stack)-1]//pop
//		if x.Creator!=nil{
//			stack=append(stack,x.Creator)
//		}
//	}
//}
//func (v *Variable) Backward() {
//	f:=v.Creator
//	if f!=nil{
//		x:=f.input
//		x.Grad=f.Back(v).Grad
//		x.Backward() //链式传播
//	}
//}

//-------Function
func NewFunction(f IFunc) Function {
	return Function{f, nil, nil}
}

type IFunc interface {
	forward(i *mat.Dense) *mat.Dense
	backward(i,dy *mat.Dense) *mat.Dense
}

type Function struct{
	iFunc       IFunc
	input,ouput *Variable
}
func (f *Function) Run(i *Variable) *Variable{
	x:=i.Data
	f.input=i

	y:=f.iFunc.forward(x)
	o:=&Variable{y,nil,f}
	f.ouput=o
	return o
}

func (f *Function) Back(i *Variable) *Variable{
	b:=f.iFunc.backward(f.input.Data,i.Grad)
	o:=&Variable{i.Data,b,f}
	return o
}

func square(x *Variable)*Variable{
	f:=NewFunction(&Square{})
	return f.Run(x)
}

type Square struct {
	Function
}
func (s *Square) forward(i *mat.Dense) *mat.Dense  {
	o:=mat.Dense{}
	o.Pow(i,2)
	return &o
}
func (s *Square) backward(i,dy *mat.Dense) *mat.Dense  {
	mul2:=func(_,_ int,v float64) float64{return v*2}
	o:=mat.Dense{}
	o.Mul(i,dy) //todo 维度问题
	o.Apply(mul2,&o)
	return &o
}

func exp(x *Variable)*Variable{
	f:=NewFunction(&Exp{})
	return f.Run(x)
}

type Exp struct {
	Function
}

func (e *Exp)forward(i *mat.Dense) *mat.Dense  {
	o:=mat.Dense{}
	o.Exp(i)
	return &o
}
func (e *Exp)backward(i,dy *mat.Dense) *mat.Dense  {
	o:=mat.Dense{}
	o.Exp(i)
	o.Mul(&o,dy)
	return &o
}


func main() {

	m := mat.NewDense(1, 1, []float64{
		0.5,
	})
	d := mat.NewDense(1,1,[]float64{
		1.0,
	})
	x := &Variable{m,d,nil}
	printDense("x", x.Data)
	//A := NewFunction(&Square{})
	//y:=A.Run(x)
	//printDense("A",y.Data)

	//check A numdiff
	//A := NewFunction(&Square{})
	//f := A.Run
	//g := numerical_diff(f,x,1e-4)
	//printDense("g",g)

	//check b numdiff
	//A := NewFunction(&Exp{})
	//f := A.Run
	//g := numerical_diff(f,x,1e-4)
	//printDense("g",g)

	//com:= func (x *Variable) *Variable {
	//	A := NewFunction(&Square{})
	//	B := NewFunction(&Exp{})
	//	C := NewFunction(&Square{})
	//	y := C.Run(B.Run(A.Run(x)))
	//	return y
	//}
	//y := com(x)
	//printDense("comdy",y.Data)

	//dy := numerical_diff(com,x,0)
	//printDense("comdy",dy)


	a := square(x)
	b := exp(a)
	y := square(b)
	printDense("y", y.Data)
	y.Grad =d
	y.Backward()
	printDense("xg",x.Grad)
	//
	//y.Grad=d
	//b.Grad=C.Back(y).Grad
	//a.Grad=B.Back(b).Grad
	//x.Grad=A.Back(a).Grad
	//printDense("g",x.Grad)
}

func LikeOnes(i *mat.Dense) *mat.Dense {
	r,c:=i.Dims()
	d := make([]float64, r*c)
	for di:= range d{
		d[di]=1
	}
	return mat.NewDense(r, c, d)
}

func LikeZeros(i *mat.Dense) *mat.Dense {
	r,c:=i.Dims()
	d := make([]float64, r*c)
	return mat.NewDense(r, c, d)
}

func numericalDiff(f func(i *Variable) *Variable,x *Variable,eps float64) * mat.Dense{
	if eps== 0 {
		eps=1E-4
	}
	add:=func(_,_ int,v float64) float64{return v+eps}
	dec:=func(_,_ int,v float64) float64{return v-eps}
	div:=func(_,_ int,v float64) float64{return v/(2*eps)}

	x0:=Variable{}
	x0.Data=&mat.Dense{}
	x0.Data.Apply(add,x.Data)
	x1:=Variable{}
	x1.Data=&mat.Dense{}
	x1.Data.Apply(dec,x.Data)

	y0,y1:=f(&x0),f(&x1)

	y0.Data.Sub(y0.Data,y1.Data)
	o:=&mat.Dense{}
	o.Apply(div,y0.Data)
	return o
}

func printDense(name str,x *mat.Dense) {
	fx := mat.Formatted(x, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("%s = %v\n",name, fx)
}

func sprintDense(x *mat.Dense) string {
	fx := mat.Formatted(x, mat.Prefix(""), mat.Squeeze())
	rst := fmt.Sprintf("%v\n", fx)
	return rst
}

