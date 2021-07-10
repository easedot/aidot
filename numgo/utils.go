package numgo

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
)

func _sumToR(x *Variable) *Variable{
	xs:=x.Shape()
	xr,xc:=xs.R,xs.C
	b:=mat.Dense{}
	y:=b.Grow(1,xc).(*mat.Dense)
	for ir:=0;ir<xr;ir++{
		w := x.Data.Slice(ir, ir+1, 0, xc).(*mat.Dense)
		for c := 0; c < xc; c++ {
			y.Set(0, c, w.At(0, c)+y.At(0, c))
		}
	}
	return &Variable{Data:y}
}
func _sumToC(x * Variable) *Variable{
	xs:=x.Shape()
	xr,xc:=xs.R,xs.C
	b:=mat.Dense{}
	y:=b.Grow(xr,1).(*mat.Dense)
	for ic:=0;ic<xc;ic++{
		w := x.Data.Slice(0, xr, ic, ic+1).(*mat.Dense)
		for r := 0; r < xr; r++ {
			y.Set(r, 0, w.At(r, 0)+y.At(r, 0))
		}
	}
	return &Variable{Data:y}
}
func _sumTo(x *Variable,s *Shape)*Variable{
	y:=x
	if s.R==1 && s.C==1{
		s:=mat.Sum(x.Data)
		y= NewVar(s)
	}
	if s.R==1 && s.C!=1{
		y=_sumToR(x)
	}
	if s.C==1 && s.R!=1{
		y=_sumToC(x)
	}
	return y
}
func _broadcastTo(x *Variable,s *Shape) *Variable{
	xs:=x.Shape()
	xr,xc:=xs.R,xs.C
	b:=mat.Dense{}
	y:=b.Grow(s.R,s.C).(*mat.Dense)
	if s.BA(xs){
		//col
		for ic:=0;ic<s.C;ic+=xc{
			w := y.Slice(0, xr, ic, ic+xc).(*mat.Dense)
			w.Copy(x.Data)
		}
		wc:=y.Slice(0,1,0,s.C)
		//row
		for ir:=0;ir<s.R;ir+=xr{
			w := y.Slice(ir, ir+xr, 0, s.C).(*mat.Dense)
			w.Copy(wc)
		}
		return &Variable{Data:y}
	} else if s.BR(xs){
		for ir:=0;ir<s.R;ir+=xr{
			w := y.Slice(ir, ir+xr, 0, xc).(*mat.Dense)
			w.Copy(x.Data)
		}
	} else if s.BC(xs){
		for ic:=0;ic<s.C;ic+=xc{
			w := y.Slice(0, xr, ic, ic+xc).(*mat.Dense)
			w.Copy(x.Data)
		}
	}
	return &Variable{Data:y}
}
func _checkBroadCast(x0s *Shape, x1s *Shape, x0 *Variable, x1 *Variable) (*Variable, *Variable) {
	if !x0s.E(x1s) {
		if x0s.B(x1s) {
			x1 = broadCastTo(x1, x0s)
		}
		if x1s.B(x0s) {
			x0 = broadCastTo(x0, x1s)
		}
	}
	return x0,x1
}
func _checkSumTo(x0s *Shape, x1s *Shape, gx0 *Variable, gx1 *Variable) (*Variable, *Variable) {
	if !x0s.E(x1s) {
		if x0s.B(x1s) {
			gx1 = sumTo(gx1, x1s)
		}
		if x1s.B(x0s) {
			gx0 = sumTo(gx0, x0s)
		}
	}
	return gx0, gx1
}

func NumericalDiff(f func(i ...*Variable) *Variable,x *Variable) * mat.Dense{
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
		log.Fatal(err)
	}

	defer os.Remove(tmpfile.Name()) // clean up

	if _, err := tmpfile.Write([]byte(content)); err != nil {
		log.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		log.Fatal(err)
	}
	os.Remove(file)
	dotCmd, err := exec.LookPath("dot")
	if err != nil {
		log.Printf("'dot' not found")
	} else {
		log.Printf("'dot' is in '%s'\n", dotCmd)
	}
	extType := filepath.Ext(file)[1:]
	cmdStr:=fmt.Sprintf("CMD: %s %s -T %s -o %s",dotCmd,tmpfile.Name(), extType,file)
	log.Printf(cmdStr)
	cmd:=exec.Command(dotCmd,tmpfile.Name(),"-T", extType,"-o",file)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err=cmd.Run()
	if err!=nil{
		log.Printf("failed to call cmd.Run(): %v",err)
	}

}
