package numgo

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
	nd "test_ai/numed"
	ut "test_ai/utils"
)

var Backprop =true




func NewArange(start, stop, step float64)*Variable{
	s:=ut.Arange(start,stop,step)
	return NewVec(s...)
}

func NewZeros(r,c int) *Variable {
	s:= make([]float64, r*c)
	return NewMat(r,c,s...)
}
//func NewOnes(r,c int) *Variable {
//	d := make([]float64, r*c)
//	for di:= range d{
//		d[di]=1
//	}
//	return NewMat(r,c,d...)
//}

func NewRand(r,c int) *Variable {
	s:=ut.Rand(r,c)
	return NewMat(r,c,s...)
}
func NewRandN(r,c int) *Variable {
	s:=ut.RandN(r,c)
	return NewMat(r,c,s...)
}
func NewRandE(r,c int) *Variable {
	s:=ut.RandE(r,c)
	return NewMat(r,c,s...)
}


func NewVar(d float64)*Variable{
	dv:=mat.NewDense(1,1,[]float64{d})
	return &Variable{Data: dv}
}

func NewVec(d... float64)*Variable{
	l:=len(d)
	dv:=mat.NewDense(1,l,d)
	return &Variable{Data: dv}
}

func NewVecInt(d... int)*Variable{
	l:=len(d)
	df:=make([]float64,l)
	for i,v :=range d{
		df[i]=float64(v)
	}
	dv:=mat.NewDense(1,l,df)
	return &Variable{Data: dv}
}

func NewMat(r,c int,d... float64)*Variable{
	dv:=mat.NewDense(r,c,d)
	return &Variable{Data: dv}
}

func CopyData(v *Variable)*Variable{
	return &Variable{Data: v.Data}
}

type Variable struct{
	Name string
	//Data *nd.NumEd
	Data *mat.Dense
	Grad *Variable
	Creator *Function
	Level int
}
func (v *Variable)At(r,c int) float64{
	return v.Data.At(r,c)
}
func (v *Variable)Grow(r,c int){
	v.Data=v.Data.Grow(r,c).(*mat.Dense)
}
func (v *Variable)GrowTo(r,c int){
	s:=v.Shape()
	v.Data=v.Data.Grow(r-s.R,c-s.C).(*mat.Dense)
}

func (v *Variable)Plot(verbose bool,file string){
	PlotDotGraph(v,verbose,file)
}

func (v *Variable)Print(name string){
	if name!=""{
		ut.PrintDense(name,v.Data)
	}else{
		ut.PrintDense(v.Name,v.Data)
	}
}
func (v *Variable)Sprint(name string) string {
	rest:=""
	if name!=""{
		rest=ut.SprintDense(name,v.Data)
	}else{
		rest=ut.SprintDense(v.Name,v.Data)
	}
	return rest
}
func (v *Variable) DataType() string{
	return fmt.Sprintf("%T",v.Data.At(0,0))
}
func (v *Variable) Shape() *nd.Shape{
	return nd.NewShape(v.Data.Dims())
}
func (v *Variable) ReShape(r,c int)*Variable{
	s:=nd.NewShape(r,c)
	return Reshape(v,s)
}

func (v *Variable) Transpose()*Variable{
	return Transpose(v)
}
func (v *Variable) T()*Variable{
	return Transpose(v)
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
		v.Grad=&Variable{Data: ut.LikeOnes(v.Data)}
	}
	seen:=make(map[*Function]bool)
	var stack []*Function
	if v.Creator!=nil{
		stack=append(stack,v.Creator)
	}
	for len(stack)>0 {
		f:=stack[len(stack)-1]
		stack=stack[:len(stack)-1]//pop
		var gys []*Variable
		for _,output:=range f.Outputs{
			gys =append(gys,output.Grad)
		}
		gxs :=f.Back(gys...)
		//这里如果有两个输入，则两个输入都会加入func列表
		for xi,x:=range f.Inputs{
			if x.Grad==nil{
				x.Grad= gxs[xi]
			}else{
				x.Grad= Add(x.Grad, gxs[xi])
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
			for _,o :=range f.Outputs{
				o.Grad=nil
			}
		}
	}
}
func (v *Variable) Var() float64{
	return v.At(0,0)
}
func (v *Variable) Mean() float64{
	xr,xc:=v.Data.Dims()
	size:=float64(xr*xc)
	trued:=0.0
	for i:=0;i<xr;i++{
		for j:=0;j<xc;j++{
			if v.At(i,j)==1{
				trued+=1
			}
		}
	}
	return trued/size
}
//Function

func NewFunction(f IFunc) Function {
	return Function{Func: f}
}

type IFunc interface {
	forward(i []*Variable) []*Variable
	backward(i,o,dy []*Variable) []*Variable
}

type Function struct{
	Func           IFunc
	Inputs, Outputs []*Variable
	Level int
}

func (f *Function) Run(ix ...*Variable) *Variable{
	outputs:=f.Func.forward(ix)
	if Backprop{
		max:= MaxLevel(ix)
		f.Level =max
		f.Inputs =ix
		for _,o:=range outputs{
			o.SetCreator(f)
		}
		f.Outputs =outputs
	}
	return outputs[0]
}

func (f *Function) Back(ig ...*Variable) []*Variable{
	//todo 这里和书中不同，特殊考虑一下，要研究对错，加法在反向传播时如果没有广播，则原封不的传递上游的Grade
	//而一开始Grade没有Creator，会造成传递的远端的计算图没有creator，plot后只有一个变量
	//for _,g :=range ig{
	//	if g.Creator==nil{
	//		g.SetCreator(f)
	//	}
	//}
	b:=f.Func.backward(f.Inputs,f.Outputs,ig)
	return b
}

// MaxLevel -----core function
func MaxLevel(ivs []*Variable) (int){
	max:=0
	for _,v:=range ivs{
		if v!=nil && v.Level>max{
			max=v.Level
		}
	}
	return max
}

func Pow(x *Variable,c int)*Variable{
	f:=NewFunction(&powFunc{C: c})
	return f.Run(x)
}
type powFunc struct {
	Function
	C int
}
func (s *powFunc) forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Pow(i[0].Data,s.C)
	return []*Variable{{Data:&o}}
}

func (s *powFunc) backward(i,o,gy []*Variable) []*Variable  {
	x:=i[0]
	c:=NewVar(float64(s.C))
	gx:= Mul(Mul(Pow(x,s.C-1),gy[0]),c)
	return []*Variable{gx}
}

func Neg(x interface{})*Variable{
	f:=NewFunction(&negFunc{})
	return f.Run(AsVar(x))
}
type negFunc struct {
	Function
}
func (e *negFunc)forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Apply(ut.NegFunc,i[0].Data)
	return [] *Variable{{Data:&o}}
}
func (e *negFunc)backward(i,o,gy []*Variable) []*Variable  {
	ngy:=Neg(gy[0])
	return [] *Variable{ngy}
}

func Add(x0,x1 interface{})*Variable{
	f:=NewFunction(&addFunc{})
	y:=f.Run(AsVar(x0), AsVar(x1))
	return y
}
type addFunc struct {
	Function
	x0s *nd.Shape
	x1s *nd.Shape
}
func (a *addFunc) forward(ix []*Variable) []*Variable {
	x0,x1:=ix[0],ix[1]
	a.x0s, a.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(a.x0s, a.x1s, x0, x1)
	o := mat.Dense{}
	o.Add(x0.Data, x1.Data)
	return []*Variable{{Data: &o}}
}
func (a *addFunc) backward(i,o,gy []*Variable) []*Variable  {
	gx0,gx1:=gy[0],gy[0]
	gx0, gx1 = _checkSumTo(a.x0s, a.x1s, gx0, gx1)
	return []*Variable{gx0, gx1}
}

func Sub(x0,x1 interface{})*Variable{
	f:=NewFunction(&subFunc{})
	y:=f.Run(AsVar(x0), AsVar(x1))
	return y
}
type subFunc struct {
	Function
	x0s *nd.Shape
	x1s *nd.Shape
}
func (a *subFunc) forward(ix []*Variable) []*Variable {
	x0,x1:=ix[0],ix[1]
	a.x0s, a.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(a.x0s, a.x1s, x0, x1)
	o:=mat.Dense{}
	o.Sub(x0.Data,x1.Data)
	return []*Variable{{Data:&o}}
}
func (a *subFunc) backward(i,o,gy []*Variable) []*Variable  {
	g:=gy[0]
	gx0,gx1:=g,Neg(g)
	gx0, gx1 = _checkSumTo(a.x0s, a.x1s, gx0, gx1)
	return []*Variable{gx0,gx1}
}


func Mul(x0,x1 interface{})*Variable{
	f:=NewFunction(&mulFunc{})
	y:=f.Run(AsVar(x0), AsVar(x1))
	return y
}
type mulFunc struct {
	Function
	x0s *nd.Shape
	x1s *nd.Shape
}
func (m *mulFunc) forward(ix[]*Variable)[]*Variable  {
	x0,x1:=ix[0],ix[1]
	m.x0s, m.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(m.x0s, m.x1s, x0, x1)
	o:=mat.Dense{}
	o.MulElem(x0.Data,x1.Data)
	return []*Variable{{Data: &o}}
}
func (m *mulFunc) backward(i,o,gy []*Variable)[]*Variable  {
	g:=gy[0]
	x0,x1:=i[0],i[1]
	gx0:= Mul(x1,g)
	gx1:= Mul(x0,g)
	gx0,gx1  = _checkSumTo(m.x0s, m.x1s, gx0, gx1)
	return []*Variable{gx0,gx1}
}

func Div(x0,x1 interface{})*Variable {
	f:=NewFunction(&divFunc{})
	y:=f.Run(AsVar(x0), AsVar(x1))
	return y
}
type divFunc struct {
	Function
	x0s *nd.Shape
	x1s *nd.Shape
}
func (d *divFunc) forward(ix[]*Variable)[]*Variable {
	x0,x1:=ix[0],ix[1]
	d.x0s, d.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(d.x0s, d.x1s, x0, x1)
	o:=mat.Dense{}
	o.DivElem(x0.Data,x1.Data)
	return []*Variable{{Data:&o}}
}

func (d *divFunc) backward(i,o,gy []*Variable)[]*Variable {
	g:=gy[0]
	x0,x1:=i[0],i[1]
	gx0 := Div(g,x1)
	gx1 := Mul(g, Div(Neg(x0), Mul(x1, x1)))
	gx0,gx1  = _checkSumTo(d.x0s, d.x1s, gx0, gx1)
	return []*Variable{gx0, gx1}
}
