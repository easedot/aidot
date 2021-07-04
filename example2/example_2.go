//package example2
//
//import (
//	"fmt"
//
//	"gonum.org/v1/gonum/mat"
//)
//
//package main
//
//import (
//"fmt"
//"gonum.org/v1/gonum/mat"
//)
//
//type str string
//
//type Variable struct{
//	Data *mat.Dense
//	Grad *mat.Dense
//}
//func NewVariable(r,c int) *Variable{
//	d := mat.NewDense(r,c,nil)
//	o := &Variable{d,nil}
//	return o
//}
//type IFunc interface {
//	forward(i *mat.Dense) *mat.Dense
//	backward(i,id *mat.Dense) *mat.Dense
//}
//func NewFunction(f IFunc) Function {
//	return Function{f, nil, nil}
//}
//type Function struct{
//	iFunc       IFunc
//	input,ouput *Variable
//}
//func (f *Function) Run(i *Variable) *Variable{
//	x:=i.Data
//	y:=f.iFunc.forward(x)
//	o:=&Variable{y,nil}
//	f.input=i
//	f.ouput=o
//	return o
//}
//
//func (f *Function) Back(i *Variable) *Variable{
//	b:=f.iFunc.backward(f.input.Data,i.Grad)
//	o:=&Variable{i.Data,b}
//	return o
//}
//
//type Square struct {
//	Function
//}
//func (s *Square) forward(i *mat.Dense) *mat.Dense  {
//	o:=mat.Dense{}
//	o.Pow(i,2)
//	return &o
//}
//func (s *Square) backward(i,id *mat.Dense) *mat.Dense  {
//	//2*x*id
//	x := i
//	mul2:=func(_,_ int,v float64) float64{return v*2}
//	o:=mat.Dense{}
//	o.Mul(x,id) //todo 维度问题
//	o.Apply(mul2,&o)
//	return &o
//}
//
//type Exp struct {
//	Function
//}
//
//func (e *Exp)forward(i *mat.Dense) *mat.Dense  {
//	o:=mat.Dense{}
//	o.Exp(i)
//	return &o
//}
//func (e *Exp)backward(i,id *mat.Dense) *mat.Dense  {
//	x:=i
//	//y=exp(x)*id
//	o:=mat.Dense{}
//	o.Exp(x)
//	r:=mat.Dense{}
//	r.Mul(&o,id)
//	return &r
//}
//
//func numerical_diff(f func(i *Variable) *Variable,x *Variable,eps float64) * mat.Dense{
//	if eps== 0 {
//		eps=1E-4
//	}
//	add:=func(_,_ int,v float64) float64{return v+eps}
//	dec:=func(_,_ int,v float64) float64{return v-eps}
//	div:=func(_,_ int,v float64) float64{return v/(2*eps)}
//
//	x0:=Variable{&mat.Dense{},&mat.Dense{}}
//	x0.Data.Apply(add,x.Data)
//	x1:=Variable{&mat.Dense{},&mat.Dense{}}
//	x1.Data.Apply(dec,x.Data)
//
//	y0,y1:=f(&x0),f(&x1)
//
//	y0.Data.Sub(y0.Data,y1.Data)
//	o:=&mat.Dense{}
//	o.Apply(div,y0.Data)
//	return o
//}
//
//func main() {
//
//	m := mat.NewDense(1, 1, []float64{
//		0.5,
//	})
//	d := mat.NewDense(1,1,[]float64{
//		1.0,
//	})
//	x := &Variable{m,d}
//	//A := NewFunction(&Square{})
//	////B := NewFunction(&Exp{})
//	////C := NewFunction(&Square{})
//	//y:=A.Run(x)
//	//printDense("A",y.Data)
//
//	com:= func (x *Variable) *Variable {
//		A := NewFunction(&Square{})
//		B := NewFunction(&Exp{})
//		C := NewFunction(&Square{})
//		y := C.Run(B.Run(A.Run(x)))
//		return y
//	}
//	////
//	//y := com(x)
//	//printDense("comdy",y.Data)
//
//	dy := numerical_diff(com,x,0)
//	printDense("comdy",dy)
//
//	//check A numdiff
//	//A := NewFunction(&Square{})
//	//f := A.Run
//	//g := numerical_diff(f,x,1e-4)
//	//printDense("g",g)
//
//	//check b numdiff
//	//A := NewFunction(&Exp{})
//	//f := A.Run
//	//g := numerical_diff(f,x,1e-4)
//	//printDense("g",g)
//
//
//	//B := NewFunction(&Exp{})
//	//C := NewFunction(&Square{})
//	//a := A.Run(x)
//	//b := B.Run(a)
//	//y = C.Run(b)
//
//	//y.Grad=d
//	//b.Grad=C.Back(y).Grad
//	//a.Grad=B.Back(b).Grad
//	//x.Grad=A.Back(a).Grad
//
//
//	//printDense("y",y.Data)
//	//
//	//printDense("dy",dy)
//	//
//	//printDense("grad",y.Grad)
//}
//
//func printDense(s str,x *mat.Dense) {
//	fx := mat.Formatted(x, mat.Prefix("    "), mat.Squeeze())
//	fmt.Printf("%s = %v\n",s, fx)
//}
//
//
