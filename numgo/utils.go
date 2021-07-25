package numgo

import (
	"fmt"
	"io/ioutil"
	logger "log"
	"os"
	"os/exec"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
	"test_ai/intset"
	nd "test_ai/numed"
	ut "test_ai/utils"
)
func AsVar(v interface{}) *Variable{
	switch v.(type) {
	case *nd.NumEd:
		return &Variable{Data: v.(*nd.NumEd)}
	case float64:
		return NewVar(v.(float64))
	case *Variable:
		return v.(*Variable)
	case int:
		return NewVar(float64(v.(int)))
	default:
		logger.Printf("input type error")
		return nil
	}
}
type condFunc func(x,y *Variable,r,c int)
func _where(x,y *Variable,cond condFunc) *Variable{
	shape:=x.Shape()
	for i:=0;i<shape.R;i++{
		for j:=0;j<shape.C;j++{
			cond(x,y,i,j)
		}
	}
	return y
}

//func _sumToR(x *Variable) *Variable{
//	xs:=x.Shape()
//	xr,xc:=xs.R,xs.C
//	b:=mat.Dense{}
//	y:=b.Grow(1,xc).(*mat.Dense)
//	for ir:=0;ir<xr;ir++{
//		w := x.Data.Slice(ir, ir+1, 0, xc).(*mat.Dense)
//		for c := 0; c < xc; c++ {
//			y.Set(0, c, w.At(0, c)+y.At(0, c))
//		}
//	}
//	return &Variable{Data:y}
//}
//func _sumToC(x * Variable) *Variable{
//	xs:=x.Shape()
//	xr,xc:=xs.R,xs.C
//	b:=mat.Dense{}
//	y:=b.Grow(xr,1).(*mat.Dense)
//	for ic:=0;ic<xc;ic++{
//		w := x.Data.Slice(0, xr, ic, ic+1).(*mat.Dense)
//		for r := 0; r < xr; r++ {
//			y.Set(r, 0, w.At(r, 0)+y.At(r, 0))
//		}
//	}
//	return &Variable{Data:y}
//}

func _sumTo(x *Variable,s *nd.Shape)*Variable{
	y:=NewMat(s.R,s.C)
	if s.R==1 && s.C==1{
		y.Data= x.Data.Sum(nil,true)
	}
	if s.R==1 && s.C!=1{
		y.Data=x.Data.Sum(0,true)
	}
	if s.C==1 && s.R!=1{
		y.Data=x.Data.Sum(1,true)
	}
	return y
}

func _logsumexp(x *Variable,axis interface{})*Variable{
	m:=Max(x,axis)
	y:=Sub(x,m)
	y=Exp(y)
	s:=_sum(y,axis,true)
	s=Log(s)
	m=Add(s,m)
	return m
}

func _maxBackwardShape(x *Variable,axis interface{})[]int{
	n:=x.Shape()
	is:=intset.IntSet{}
	en:=[]int{n.R,n.C}
	if axis==nil{
		r:=ut.ArangeInt(0,2,1)
		is.AddAll(r...)
	}else{
		switch axis.(type) {
		case int:
			is.Add(axis.(int))
		case []int:
			is.AddAll(axis.([]int)...)
		}
	}
	var s []int
	for i,v:=range en{
		if !is.Has(i){
			s=append(s,v)
		}else{
			s=append(s,1)
		}
	}
	return s
}

func _eyes(n int)*Variable{
	d:=nd.NewEyes(n)
	return &Variable{Data: d}
}

func _sum(x *Variable,axis interface{},keepDims bool)*Variable{
	y:= x.Data.Sum(axis,keepDims)
	return &Variable{Data:y}
}
func _min(x *Variable,axis interface{},keepDims bool)*Variable{
	y:= x.Data.Min(axis,keepDims)
	return &Variable{Data:y}
}

func AgrMax(x *Variable,axis int,keepDims bool)*Variable{
	return _agrMax(x,axis,keepDims)
}

func _agrMax(x *Variable,axis int,keepDims bool)*Variable{
	y:=x.Data.ArgMax(axis,keepDims)
	return &Variable{Data:y}
}
func _max(x *Variable,axis interface{},keepDims bool)*Variable{
	y:= x.Data.Max(axis, keepDims)
	return &Variable{Data:y}
}

type RowColFunc func(x,y,iy *mat.Dense,r,c,idx int)

func OneHot(x *Variable,rs []int )*Variable{
	m:=x.Data.Rows(rs...)
	return &Variable{Data: m}
}
func SelRow(x *mat.Dense,rs...int)*mat.Dense{
	mr:=len(rs)
	_,xc:=x.Dims()
	m:=mat.NewDense(mr,xc,nil)
	for i,r :=range rs{
		from := x.Slice(r, r+1, 0, xc).(*mat.Dense)
		mw:= m.Slice(i,i+1,0,xc).(*mat.Dense)
		mw.Copy(from)
	}
	return m
}

func ColData(x *mat.Dense,c int) []float64{
	xr,_:=x.Dims()
	m:=mat.NewDense(xr,1,nil)
	for i:=0;i<xr;i++{
		m.Set(i,0,x.At(i,c))
	}
	return m.RawMatrix().Data
}

func SelRowCol(x *mat.Dense,rs...int)*mat.Dense{
	mr:=len(rs)
	m:=mat.NewDense(mr,1,nil)
	for i,c :=range rs{
		m.Set(i,0,x.At(i,c))
	}
	return m
}

func _tranposeTo(x *Variable)*Variable{
	y:=x.Data.T()
	return &Variable{Data:y}
}


func Ravel(x*mat.Dense)[]float64{
	return x.RawMatrix().Data
}

func Cross(x,y *mat.Dense)*mat.Dense{
	xr,_:=x.Dims()
	m:=mat.NewDense(xr,2,nil)
	for i:=0;i<xr;i++{
		m.Set(i,0,x.At(0,i))
		m.Set(i,1,y.At(0,i))
	}
	return m
}

func MeshGrid(x,y *Variable)(*Variable,*Variable){
	xs:=x.Shape()
	ys:=y.Shape()
	sp:=nd.NewShape(ys.C,xs.C)
	y=y.T()
	xm:=_broadcastTo(x,sp)
	ym:=_broadcastTo(y,sp)
	return xm,ym
}

func _broadcastTo(x *Variable,s *nd.Shape) *Variable{
	y:=x.Data.BroadcastTo(s)
	return &Variable{Data:y}
}
func _checkBroadCast(x0s *nd.Shape, x1s *nd.Shape, x0 *Variable, x1 *Variable) (*Variable, *Variable) {
	if !x0s.E(x1s) {
		if x0s.B(x1s) {
			x1 = BroadCastTo(x1, x0s)
		}
		if x1s.B(x0s) {
			x0 = BroadCastTo(x0, x1s)
		}
	}
	return x0,x1
}
func _checkSumToV(gx0 *Variable, gx1 *Variable) (*Variable, *Variable) {
	x0s,x1s:=gx0.Shape(),gx1.Shape()
	x0,x1:=_checkSumTo(x0s,x1s,gx0,gx1)
	return x0,x1
}
func _checkSumTo(x0s *nd.Shape, x1s *nd.Shape, gx0 *Variable, gx1 *Variable) (*Variable, *Variable) {
	if !x0s.E(x1s) {
		if x0s.B(x1s) {//x1 做过broadcast
			gx1 = SumTo(gx1, x1s)
		}
		if x1s.B(x0s) {//x0 做过broadcast
			gx0 = SumTo(gx0, x0s)
		}
	}
	return gx0, gx1
}

func NumericalDiff(f func(i *Variable) *Variable,x *Variable) * Variable {
	eps:=1e-6
	grad:=  nd.LikeZeros(x.Data)
	xs:=x.Shape()
	for i:=0;i<xs.R;i++ {
		for j:=0;j<xs.C;j++{
			tempV := x.Data.At(i, j)
			y3:=f(x)
			x.Data.Set(i,j, tempV+eps)
			y1:=f(x)
			x.Data.Set(i,j, tempV-eps)
			y2:=f(x)
			sub := Sub(y1, y2)
			diff:= Sum(sub)
			diffV := diff.Data.At(0, 0)
			g := diffV / (2.0 * eps)
			//这里处理max，min的特殊情况，如果3-eps,但是y没有减小，因为max返回最大值了
			if nd.Equal(y1.Data,y3.Data,1e-8) || nd.Equal(y2.Data,y3.Data,1e-8){
			//if mat.EqualApprox(y1.Data,y3.Data,1e-8) || mat.EqualApprox(y2.Data,y3.Data,1e-8){
				g = diffV / (1.0 * eps)
			}

			grad.Set(i,j, g)

			x.Data.Set(i,j,tempV)
		}
	}
	return &Variable{Data:grad}
}
func dotVar(v *Variable,verbose bool) string{
	name:=v.Name
	if verbose && v.Data!=nil{
		if name!=""{
			name+=":"
		}
		vs:=v.Shape()
		r,c:=vs.R,vs.C
		name=fmt.Sprintf("%s (%d,%d) %s",name,r,c,v.DataType())
	}
	dotVar:=fmt.Sprintf(" \"%p\" [label=\"%s\", color=orange,style=filled]\n",v,name)
	return dotVar
}
func dotFunc(f *Function) string{
	dotFunc:=fmt.Sprintf("\"%p\" [label=\"%T\" color=lightblue,style=filled,shape=box]\n",f,f.Func)
	for _,i :=range f.Inputs{
		dotFunc+=fmt.Sprintf("\"%p\"->\"%p\"\n",i,f)
	}
	for _,o :=range f.Outputs{
		dotFunc+=fmt.Sprintf("\"%p\"->\"%p\"\n",f,o)
	}
	return dotFunc
}
func getDotGraph(v *Variable,verbose bool)string{
	txt:=""
	seen:=make(map[*Function]bool)
	txt+=dotVar(v,verbose)
	var stack []*Function
	if v.Creator!=nil{
		stack=append(stack,v.Creator)
	}
	for len(stack)>0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1] //pop
		txt+=dotFunc(f)

		//这里如果有两个输入，则两个输入都会加入func列表
		for _, x := range f.Inputs {
			txt+=dotVar(x,verbose)
			if x.Creator != nil && !seen[x.Creator] {
				seen[x.Creator] = true
				stack = append(stack, x.Creator)
			}
		}
	}
	return fmt.Sprintf("digraph g {\n%s}\n",txt)
}

func PlotDotGraph(v *Variable,verbose bool,file string){
	content:=getDotGraph(v,verbose)
	tmpfile, err := ioutil.TempFile("", "tmp_graph.dot")
	if err != nil {
		logger.Fatal(err)
	}

	defer os.Remove(tmpfile.Name()) // clean up

	if _, err := tmpfile.Write([]byte(content)); err != nil {
		logger.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		logger.Fatal(err)
	}
	os.Remove(file)
	dotCmd, err := exec.LookPath("dot")
	if err != nil {
		logger.Printf("'dot' not found")
	} else {
		logger.Printf("'dot' is in '%s'\n", dotCmd)
	}
	extType := filepath.Ext(file)[1:]
	cmdStr:=fmt.Sprintf("CMD: %s %s -T %s -o %s",dotCmd,tmpfile.Name(), extType,file)
	logger.Printf(cmdStr)
	cmd:=exec.Command(dotCmd,tmpfile.Name(),"-T", extType,"-o",file)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err=cmd.Run()
	if err!=nil{
		logger.Printf("failed to call cmd.Run(): %v",err)
	}

}
