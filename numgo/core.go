package numgo

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
	ut "test_ai/utils"
)

const Backprop =true

func NewShape(r,c int)*Shape{
	return &Shape{r,c}
}
type Shape struct {
	R,C int
}

func (s *Shape) E(i *Shape) bool{
	return s.R==i.R && s.C==i.C
}
func (s *Shape) G(i *Shape) bool{
	return s.R>i.R && s.C>=i.C||s.C>i.C && s.R>=i.R
}
func (s *Shape) L(i *Shape) bool{
	return s.R<i.R && s.C<=i.C||s.C<i.C && s.R<=i.R
}
func (s *Shape) X(i *Shape) bool{
	return s.R<i.R && s.C>i.C||s.C<i.C && s.R>i.R
}
func (s *Shape) B(i *Shape) bool{
	return s.BA(i)||s.BR(i)||s.BC(i)
}
func (s *Shape) BA(i *Shape) bool{
	//return s.R%i.R==0 && s.C%i.C==0
	return i.R==1 && i.C==1
}

func (s *Shape) BR(i *Shape) bool{
	return s.R%i.R==0 && s.C==i.C
}
func (s *Shape) BC(i *Shape) bool{
	return s.C%i.C==0 && s.R==i.R
}



func NewArange(start, stop, step float64)*Variable{
	s:=ut.Arange(0,10.,0.5)
	return NewVec(s...)
}

func NewZeros(r,c int) *Variable {
	s:= make([]float64, r*c)
	return NewMat(r,c,s...)
}
func NewOnes(r,c int) *Variable {
	d := make([]float64, r*c)
	for di:= range d{
		d[di]=1
	}
	return NewMat(r,c,d...)
}

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
func NewMat(r,c int,d... float64)*Variable{
	dv:=mat.NewDense(r,c,d)
	return &Variable{Data: dv}
}

func CopyData(v *Variable)*Variable{
	return &Variable{Data: v.Data}
}

type Variable struct{
	Name string
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
func (v *Variable) Shape() *Shape{
	return NewShape(v.Data.Dims())
}
func (v *Variable) ReShape(r,c int)*Variable{
	s:=NewShape(r,c)
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

