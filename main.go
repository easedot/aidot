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
		idx=append(idx,x.Grad.Data)
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
		v:=&Variable{Grad:&Variable{Data: d}}
		ivx=append(ivx,v)
	}
	return ivx
}

type Variable struct{
	Name string
	Data *mat.Dense
	Grad *Variable
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
		v.Grad=&Variable{Data: LikeOnes(v.Data)}
	}
	seen:=make(map[*Function]bool)
	stack:=[]*Function{v.Creator}
	for len(stack)>0 {
		f:=stack[len(stack)-1]
		stack=stack[:len(stack)-1]//pop
		var gys []*Variable
		for _,output:=range f.outputs{
			gys =append(gys,output.Grad)
		}
		gxs :=f.Back(gys...)
		//这里如果有两个输入，则两个输入都会加入func列表
		for xi,x:=range f.inputs{
			if x.Grad==nil{
				x.Grad= gxs[xi]
			}else{
				x.Grad=add(x.Grad, gxs[xi])
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
	backward(i,dy []*Variable) []*Variable
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
	b:=f.iFunc.backward(f.inputs,ig)
	return b
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

func (s *Pow) backward(i,dy []*Variable) []*Variable  {
	mul2:=func(_,_ int,v float64) float64{return v*float64(s.C)}
	x:=i[0]
	o:=mul(pow(x,s.C-1),dy[0])
	//o.Apply(mul2,&o)
	//todo Apply这种如何处理，需要后续考虑,暂时原始处理，计算图断掉了！！！
	o.Data.Apply(mul2,o.Data)
	return []*Variable{o}
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

func (e *Exp)backward(i, gy []*Variable) []*Variable {
	o:=mul(exp(i[0]), gy[0])
	return [] *Variable{o}
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
func (e *Neg)backward(i,gy []*Variable) []*Variable  {
	ngy:=neg(gy[0])
	return [] *Variable{ngy}
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
func (a *Sub) backward(i,gy []*Variable) []*Variable  {
	ngy:=neg(gy[0])
	return []*Variable{gy[0],ngy}
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
func (d *Div) backward(i,gy []*Variable)[]*Variable {
	x0,x1:=i[0],i[1]
	gx0 :=div(gy[0],x1)
	gx1 :=mul(gy[0], div(neg(x0), pow(x1, 2)))
	return []*Variable{gx0, gx1}
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
	return []*mat.Dense{&o}
}
func (a *Add) backward(i,gy []*Variable) []*Variable  {
	return []*Variable{gy[0],gy[0]}
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
func (m *Mul) backward(i,gy []*Variable)[]*Variable  {
	x0,x1:=i[0],i[1]
	return []*Variable{mul(gy[0],x1),mul(gy[0],x0)}
}

func main() {
	m := mat.NewDense(1, 1, []float64{
		3,
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

	com:= func (x...*Variable) *Variable {
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

