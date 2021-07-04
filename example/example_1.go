//package example1
//
//import (
//	"fmt"
//
//	"gonum.org/v1/gonum/mat"
//)
//
//type Variable struct{
//	Data *mat.Dense
//}
//
//type iFunc interface {
//	forward(i *mat.Dense) *mat.Dense
//}
//type Function struct{
//	iFunc       iFunc
//}
//func (f *Function) Run(i *Variable) *Variable {
//	x:=i.Data
//	y:=f.iFunc.forward(x)
//	o:= Variable{y}
//	return &o
//}
//
//type Squre struct {
//	Function
//}
//func (s *Squre) forward(i *mat.Dense) *mat.Dense  {
//	o:=mat.Dense{}
//	o.Pow(i,2)
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
//
//func main() {
//	// This example copies the second column of a matrix into col, allocating a new slice of float64.
//	m := mat.NewDense(3, 3, []float64{
//		2.0, 9.0, 3.0,
//		4.5, 6.7, 8.0,
//		1.2, 3.0, 6.0,
//	})
//	//todo 03
//	//A := Function{&Squre{}}
//	//B := Function{&Exp{}}
//	//C := Function{&Squre{}}
//	//
//	//x := Variable{m}
//	//a := A.Run(&x)
//	//b := B.Run(a)
//	//y := C.Run(b)
//
//	A := Function{&Squre{}}
//	B := Function{&Exp{}}
//	C := Function{&Squre{}}
//
//	x := Variable{m}
//	a := A.Run(&x)
//	b := B.Run(a)
//	y := C.Run(b)
//
//	fx := mat.Formatted(x.Data, mat.Prefix("    "), mat.Squeeze())
//	fmt.Printf("x = %v\n", fx)
//
//	fy := mat.Formatted(y.Data, mat.Prefix("    "), mat.Squeeze())
//	fmt.Printf("y = %v\n", fy)
//}

