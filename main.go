package main

import "C"
import (
	"fmt"
	"io/ioutil"
	"log"
	"os/exec"
	"path/filepath"
	"sort"

	"gonum.org/v1/gonum/mat"
	//mat "github.com/nlpodyssey/spago/pkg/mat32"

)

type str string
const Backprop =true

var negFunc=func(_,_ int,v float64) float64{return -v}

type Variable struct{
	Name string
	Data *mat.Dense
	Grad *Variable
	Creator *Function
	Level int
}
func (v *Variable) DataType() string{
	return fmt.Sprintf("%T",v.Data.At(0,0))
}
func (v *Variable) Shape() (int,int){
	return v.Data.Dims()
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
	forward(i []*Variable) []*Variable
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
	outputs:=f.iFunc.forward(ix)
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
func (s *Pow) forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Pow(i[0].Data,s.C)
	return []*Variable{{Data:&o}}
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
func (e *Exp)forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Exp(i[0].Data)
	return [] *Variable{{Data:&o}}
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
func (e *Neg)forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Apply(negFunc,i[0].Data)
	return [] *Variable{{Data:&o}}
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
func (a *Sub) forward(ix []*Variable) []*Variable {
	o:=mat.Dense{}
	o.Sub(ix[0].Data,ix[1].Data)
	return []*Variable{{Data:&o}}
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

func (d *Div) forward(ix[]*Variable)[]*Variable {
	o:=mat.Dense{}
	o.DivElem(ix[0].Data,ix[1].Data)
	return []*Variable{{Data:&o}}
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
func (a *Add) forward(ix []*Variable) []*Variable {
	o:=mat.Dense{}
	o.Add(ix[0].Data,ix[1].Data)
	return []*Variable{{Data:&o}}
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

func (m *Mul) forward(ix[]*Variable)[]*Variable  {
	o:=mat.Dense{}
	o.MulElem(ix[0].Data,ix[1].Data)
	return []*Variable{{Data: &o}}
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
	x.Name="x"
	print("x:",dotVar(x,true))
	y1 := pow(x,2)
	y1.Name="y"
	print("y:",dotVar(y1,true))
	print("func:",dotFunc(y1.Creator))
	print(getDotGraph(y1,true))
	plotDotGraph(y1,true,"/Users/haihui/Downloads/pow.png")
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

func dotVar(v *Variable,verbose bool) string{
	name:=v.Name
	if verbose && v.Data!=nil{
		if name!=""{
			name+=":"
		}
		r,c:=v.Shape()
		name=fmt.Sprintf("%s (%d,%d) %s",name,r,c,v.DataType())
	}
	dotVar:=fmt.Sprintf(" \"%p\" [label=\"%s\", color=orange,style=filled]\n",v,name)
	return dotVar
}
func dotFunc(f *Function) string{
	dotFunc:=fmt.Sprintf("\"%p\" [label=\"%T\" color=lightblue,style=filled,shape=box]\n",f,f.iFunc)
	for _,i :=range f.inputs{
		dotFunc+=fmt.Sprintf("\"%p\"->\"%p\"\n",i,f)
	}
	for _,o :=range f.outputs{
		dotFunc+=fmt.Sprintf("\"%p\"->\"%p\"\n",f,o)
	}
	return dotFunc
}
func getDotGraph(v *Variable,verbose bool)string{
	txt:=""
	seen:=make(map[*Function]bool)
	stack:=[]*Function{v.Creator}
	txt+=dotVar(v,verbose)
	for len(stack)>0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1] //pop
		txt+=dotFunc(f)

		//这里如果有两个输入，则两个输入都会加入func列表
		for _, x := range f.inputs {
			txt+=dotVar(x,verbose)
			if x.Creator != nil && !seen[x.Creator] {
				seen[x.Creator] = true
				stack = append(stack, x.Creator)
			}
		}
	}
	return fmt.Sprintf("digraph g {\n%s}\n",txt)
}

func plotDotGraph(v *Variable,verbose bool,file string){
	content:=getDotGraph(v,verbose)
	tmpfile, err := ioutil.TempFile("", "tmp_graph.dot")
	if err != nil {
		log.Fatal(err)
	}

	//defer os.Remove(tmpfile.Name()) // clean up

	if _, err := tmpfile.Write([]byte(content)); err != nil {
		log.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		log.Fatal(err)
	}

	path, err := exec.LookPath("dot")
	if err != nil {
		log.Printf("'dot' not found")
	} else {
		log.Printf("'dot' is in '%s'\n", path)
	}
	log.Print(tmpfile.Name())
	fileExt := filepath.Ext(file)
	cmd:=fmt.Sprintf(" %s %s -T %s -o %s",path,tmpfile.Name(),fileExt[1:],file)
	log.Printf("cmd:%s",cmd)
	exec.Command(cmd)
}