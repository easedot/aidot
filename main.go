package main

import "C"
import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
	//mat "github.com/nlpodyssey/spago/pkg/mat32"

)

type str string
const Backprop =true

var negFunc=func(_,_ int,v float64) float64{return -v}

func v2d(ix []*Variable)[]*mat.Dense{
	var idx []*mat.Dense
	for _,x:=range ix{
		idx=append(idx,x.Data)
	}
	return idx
}
func v2g(ix []*Variable)[]*mat.Dense{
	var idx []*mat.Dense
	for _,x:=range ix{
		idx=append(idx,x.Grad)
	}
	return idx
}

func d2v(id []*mat.Dense)[]*Variable{
	var ivx []*Variable
	for _,d:=range id{
		v:=&Variable{Data:d}
		ivx=append(ivx,v)
	}
	return ivx
}
func g2v(id []*mat.Dense)[]*Variable{
	var ivx []*Variable
	for _,d:=range id{
		v:=&Variable{Grad:d}
		ivx=append(ivx,v)
	}
	return ivx
}

type Variable struct{
	Data *mat.Dense
	Grad *mat.Dense
	Creator *Function
	Level int
}

func (v *Variable) ClearGrade()  {
	v.Grad=nil
}
func (v *Variable) SetCreator(f*Function)  {
	v.Creator = f
	v.Level = f.Level+1
}
func (v *Variable) Backward(retainGrad bool) {
	if v.Grad==nil{
		v.Grad=LikeOnes(v.Data)
	}
	seen:=make(map[*Function]bool)
	stack:=[]*Function{v.Creator}
	for len(stack)>0 {
		f:=stack[len(stack)-1]
		stack=stack[:len(stack)-1]//pop
		y:=f.outputs
		gys:=f.Back(y...)
		//这里如果有两个输入，则两个输入都会加入func列表
		for xi,x:=range f.inputs{
			if x.Grad==nil{
				x.Grad=gys[xi].Grad
			}else{
				x.Grad.Add(x.Grad,gys[xi].Grad)
			}
			if x.Creator!=nil && !seen[x.Creator]{
				seen[x.Creator]=true
				stack=append(stack,x.Creator)
				sort.Slice(stack, func(i, j int) bool {
					return stack[i].Level<stack[j].Level
				})
			}
		}
		if !retainGrad{
			for _,o :=range f.outputs{
				o.Grad=nil
			}
		}
	}
}

func NewFunction(f IFunc) Function {
	return Function{iFunc: f}
}

type IFunc interface {
	forward(i []*mat.Dense) []*mat.Dense
	backward(i,dy []*mat.Dense) []*mat.Dense
}

type Function struct{
	iFunc           IFunc
	inputs, outputs []*Variable
	Level int
}

func FindMaxAndMin(ivs []*Variable) (int,int){
	max,min:=0,0
	for _,v:=range ivs{
		if v.Level>max{
			max=v.Level
		}
		if v.Level<min{
			min=v.Level
		}
	}
	return max,min
}
func (f *Function) Run(ix ...*Variable) *Variable{
	idx:=v2d(ix)
	y:=f.iFunc.forward(idx)

	var outputs []*Variable
	for _,oy:=range y{
		o:=Variable{Data:oy}
		outputs=append(outputs,&o)
	}
	if Backprop{
		max,_:=FindMaxAndMin(ix)
		f.Level =max
		f.inputs =ix
		for _,o:=range outputs{
			o.SetCreator(f)
		}
		f.outputs =outputs
	}
	return outputs[0]
}

func (f *Function) Back(ig ...*Variable) []*Variable{
	idx:=v2d(f.inputs)
	igx:=v2g(ig)
	b:=f.iFunc.backward(idx,igx)
	os:=g2v(b)
	return os
}

func pow(x *Variable,c int)*Variable{
	f:=NewFunction(&Pow{C:c})
	return f.Run(x)
}
type Pow struct {
	Function
	C int
}
func (s *Pow) forward(i []*mat.Dense) []*mat.Dense  {
	o:=mat.Dense{}
	o.Pow(i[0],s.C)
	return []*mat.Dense{&o}
}

func (s *Pow) backward(i,dy []*mat.Dense) []*mat.Dense  {
	mul2:=func(_,_ int,v float64) float64{return v*float64(s.C)}
	x:=i[0]
	o:=mat.Dense{}
	o.Pow(x,s.C-1)
	o.MulElem(&o,dy[0]) //todo 维度问题
	o.Apply(mul2,&o)
	return []*mat.Dense{&o}
}

func exp(x ...*Variable)*Variable{
	f:=NewFunction(&Exp{})
	return f.Run(x...)
}
type Exp struct {
	Function
}
func (e *Exp)forward(i []*mat.Dense) []*mat.Dense  {
	o:=mat.Dense{}
	o.Exp(i[0])
	return [] *mat.Dense{&o}
}
func (e *Exp)backward(i,dy []*mat.Dense) []*mat.Dense  {
	o:=mat.Dense{}
	o.Exp(i[0])
	o.MulElem(&o,dy[0])
	return [] *mat.Dense{&o}
}

func neg(x *Variable)*Variable{
	f:=NewFunction(&Neg{})
	return f.Run(x)
}
type Neg struct {
	Function
}
func (e *Neg)forward(i []*mat.Dense) []*mat.Dense  {
	o:=mat.Dense{}
	o.Apply(negFunc,i[0])
	return [] *mat.Dense{&o}
}
func (e *Neg)backward(i,dy []*mat.Dense) []*mat.Dense  {
	o:=mat.Dense{}
	o.Apply(negFunc,dy[0])
	return [] *mat.Dense{&o}
}


func add(x0,x1 *Variable)*Variable{
	f:=NewFunction(&Add{})
	y:=f.Run(x0,x1)
	return y
}
type Add struct {
	Function
}
func (a *Add) forward(ix []*mat.Dense) []*mat.Dense {
	o:=mat.Dense{}
	o.Add(ix[0],ix[1])
	os:=[]*mat.Dense{&o}
	return os
}
func (a *Add) backward(i,gy []*mat.Dense) []*mat.Dense  {
	return []*mat.Dense{gy[0],gy[0]}
}

func sub(x0,x1 *Variable)*Variable{
	f:=NewFunction(&Sub{})
	y:=f.Run(x0,x1)
	return y
}
type Sub struct {
	Function
}
func (a *Sub) forward(ix []*mat.Dense) []*mat.Dense {
	o:=mat.Dense{}
	o.Sub(ix[0],ix[1])
	os:=[]*mat.Dense{&o}
	return os
}
func (a *Sub) backward(i,gy []*mat.Dense) []*mat.Dense  {
	ng:=mat.Dense{}
	ng.Apply(negFunc,gy[0])
	return []*mat.Dense{gy[0],&ng}
}



func div(x0,x1 *Variable)*Variable {
	f:=NewFunction(&Div{})
	y:=f.Run(x0,x1)
	return y
}
type Div struct {
	Function
}

func (d *Div) forward(ix[]*mat.Dense)[]*mat.Dense {
	o:=mat.Dense{}
	o.DivElem(ix[0],ix[1])
	return []*mat.Dense{&o}
}
func (d *Div) backward(i,gy []*mat.Dense)[]*mat.Dense {
	x0,x1:=i[0],i[1]
	gy0:=mat.Dense{}
	gy0.DivElem(gy[0],x1)
	gy1:=mat.Dense{}
	negX0:=mat.Dense{}
	negX0.Apply(negFunc,x0)
	powX1:=mat.Dense{}
	powX1.Pow(x1,2)
	divX:=mat.Dense{}
	divX.DivElem(&negX0,&powX1)
	gy1.MulElem(gy[0],&divX)
	return []*mat.Dense{&gy0,&gy1}
}

func mul(x0,x1 *Variable)*Variable{
	f:=NewFunction(&Mul{})
	y:=f.Run(x0,x1)
	return y
}
type Mul struct {
	Function
}

func (m *Mul) forward(ix[]*mat.Dense)[]*mat.Dense  {
	o:=mat.Dense{}
	o.MulElem(ix[0],ix[1])
	return []*mat.Dense{&o}
}
func (m *Mul) backward(i,gy []*mat.Dense)[]*mat.Dense  {
	x0,x1:=i[0],i[1]
	gy0:=mat.Dense{}
	gy0.MulElem(gy[0],x1)
	gy1:=mat.Dense{}
	gy1.MulElem(gy[0],x0)
	return []*mat.Dense{&gy0,&gy1}
}

func main() {
	m := mat.NewDense(1, 1, []float64{
		0.5,
	})

	//m := mat.NewDense(1, 1, []float64{
	//	2.0,
	//})
	//d := mat.NewDense(1,1,[]float64{
	//	3.0,
	//})
	x := &Variable{Data:m}
	printDense("x", x.Data)

	//y := pow(x,2)
	//printDense("y",y.Data) //4

	////check A numdiff
	//A := NewFunction(&Pow{C:2})
	//f := A.Run
	//g := numericalDiff(f,x,)
	//printDense("g",g)

	////check b numdiff
	//A := NewFunction(&Exp{})
	//y:=A.Run(x)
	//y.Backward(false)
	//f := A.Run
	//g := numericalDiff(f,x)
	//printDense("y",y.Data)
	//printDense("g",g)

	com:= func (x...*Variable) *Variable {
		a := pow(x[0],2)
		b := exp(a)
		y := pow(b,2)
		return y
	}
	y := com(x)
	y.Backward(true)
	printDense("yg",x.Grad)
	dy := numericalDiff(com,x)
	printDense("dg",dy)

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

func numericalDiff(f func(i ...*Variable) *Variable,x *Variable) * mat.Dense{
	eps:=1E-4
	addF:=func(_,_ int,v float64) float64{return v+eps}
	decF:=func(_,_ int,v float64) float64{return v-eps}
	divF:=func(_,_ int,v float64) float64{return v/(2*eps)}

	x0:=Variable{Data: &mat.Dense{}}
	x0.Data.Apply(decF,x.Data)

	x1:=Variable{Data: &mat.Dense{}}
	x1.Data.Apply(addF,x.Data)

	y0,y1:=f(&x0),f(&x1)

	o:=&mat.Dense{}
	o.Sub(y1.Data,y0.Data)
	o.Apply(divF,o)
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

