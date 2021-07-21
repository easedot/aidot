package utils

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/plotter"
)
var NegFunc=func(_,_ int,v float64) float64{return -v}
var SinFunc=func(_,_ int,v float64) float64{return math.Sin(v)}
var CosFunc=func(_,_ int,v float64) float64{return math.Cos(v)}
var TanghFunc=func(_,_ int,v float64) float64{return math.Tanh(v)}
var SqrtFunc=func(_,_ int,v float64) float64{return math.Sqrt(v)}
var ExpFunc=func(_,_ int,v float64) float64{return math.Exp(v)}
var LogFunc=func(_,_ int,v float64) float64{return math.Log(v)}
//func Rand(){
//	rand.Float64()
//	rand.Perm(10)
//	//rand.Seed()
//}

type filterFunc func(i int,v float64) bool


func Map(x []float64,f filterFunc) []float64{
	var r []float64
	for i,v:=range x{
		if f(i,v){
			r=append(r,v)
		}
	}
	return r
}
func MapIn(x []float64,f filterFunc)[]float64{
	x2:=x
	x1:=x[:0]
	for i,v:=range x2{
		if f(i,v){
			x1=append(x1,v)
		}
	}
	return x1
}

func MapSlice(x[]float64,s...int)[]float64{
	f:= func(i int,v float64)bool{
		for _,vv:=range s{
			if i==vv{
				return true
			}
		}
		return false
	}
	return Map(x,f)
}
func MapSliceIn(x[]float64,s...int)[]float64{
	f:= func(i int,v float64)bool{
		for _,vv:=range s{
			if i==vv{
				return true
			}
		}
		return false
	}
	x=MapIn(x,f)
	return x
}

func Rand(r,c int)[]float64{
	size:=r*c
	//ra:=rand.New(rand.NewSource(0))
	ra:=rand.New(rand.NewSource(time.Now().UnixNano()))
	s:=make([]float64,size,size)
	for i :=range s{
		s[i]=ra.Float64()
	}
	return s
}
func RandN(r,c int)[]float64{
	size:=r*c
	//ra:=rand.New(rand.NewSource(0))
	ra:=rand.New(rand.NewSource(time.Now().UnixNano()))
	s:=make([]float64,size,size)
	for i :=range s{
		s[i]=ra.NormFloat64()
	}
	return s
}

func RandE(r,c int)[]float64{
	size:=r*c
	//ra:=rand.New(rand.NewSource(0))
	ra:=rand.New(rand.NewSource(time.Now().UnixNano()))
	s:=make([]float64,size,size)
	for i :=range s{
		s[i]=ra.ExpFloat64()
	}
	return s
}
func Option(v,d float64) float64{
	if v==0{
		return 	d
	}else{
		return v
	}
}
func Arange(start, stop, step float64) []float64 {
	//start= Option(start,0)
	//start= Option(step,1)
	N := int(math.Ceil((stop - start) / step));
	rnge := make([]float64, N, N)
	i := 0
	for x := start; x < stop; x += step {
		rnge[i] = x;
		i += 1
	}
	return rnge
}
func ArangeInt(start, stop, step int) []int {
	//start= Option(start,0)
	//start= Option(step,1)
	N := int(math.Ceil(float64((stop - start) / step)));
	rnge := make([]int, N, N)
	i := 0
	for x := start; x < stop; x += step {
		rnge[i] = x;
		i += 1
	}
	return rnge
}

func Eyes(n int) *mat.Dense{
	d:=mat.NewDense(n,n,nil)
	for i:=0;i<n;i++{
		d.Set(i,i,1)
	}
	return d
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

func DensToPlot(x,y *mat.Dense)[]*plotter.XYs{
	ptsList:=[]*plotter.XYs{}
	xr,xc:=x.Dims()
	for i:=0;i<xr;i++{
		pts := make(plotter.XYs, xc)
		for j:=0;j<xc;j++{
			pts[j].Y=y.At(i,j)
			pts[j].X=x.At(i,j)
		}
		ptsList=append(ptsList,&pts)
	}
	return ptsList
}

func PrintDense(name string,x *mat.Dense) {
	ds:=SprintDense(name,x)
	fmt.Printf("%s\n", ds)
}

func SprintDense(name string,x *mat.Dense) string {
	rst:=""
	if name==""{
		fx := mat.Formatted(x, mat.Prefix("  "), mat.Squeeze())
		rst = fmt.Sprintf("%v ", fx)
	}else{
		fx := mat.Formatted(x, mat.Prefix("      "), mat.Squeeze())
		rst = fmt.Sprintf("%3s = %v ",name, fx)
	}
	return rst
}

func Flatten(f [][]float64) (r, c int, d []float64) {
	r = len(f)
	if r == 0 {
		panic("bad test: no row")
	}
	c = len(f[0])
	d = make([]float64, 0, r*c)
	for _, row := range f {
		if len(row) != c {
			panic("bad test: ragged inputs")
		}
		d = append(d, row...)
	}
	return r, c, d
}

func unflatten(r, c int, d []float64) [][]float64 {
	m := make([][]float64, r)
	for i := 0; i < r; i++ {
		m[i] = d[i*c : (i+1)*c]
	}
	return m
}

// eye returns a new identity matrix of size n×n.
func eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return mat.NewDense(n, n, d)
}

func SelRowInt(x []int,rs...int)[]int{
	mr:=len(rs)
	m:=make([]int,mr)
	for i,r :=range rs{
		m[i]=x[r]
	}
	return m
}

func MinFloat64Slice(v []float64)float64{
	sort.Float64Slice(v).Sort()
	return v[0]
}

func MaxFloat64Slice(v []float64)float64{
	sort.Float64Slice(v).Sort()
	return v[len(v)-1]
}
// and MinMaxFloat64 version
func MinMaxFloat64Slice(v []float64)(min float64,max float64){
	sort.Float64Slice(v).Sort()
	return v[0],v[len(v)-1]
}

//func MeshGrid(x,y []float64)(xx,yy[]float64){
//	yr,xr:=len(x),len(y)
//	g:=make([]float64,)
//	return g
//}

func CrossSlice(x,y []float64)[][]float64{
	xr:=len(x)
	r:=make([][]float64,xr)
	for i:=0;i<xr;i++{
		r[i]=[]float64{x[i],y[i]}
	}
	return r
}

//t:=ng.NewVar(1)
//t.Print("x")
//t.GrowTo(3,3)
//t.Print("tg")
//m := mat.NewDense(2, 3, []float64{
//	1,2,3,
//	3,2,1,
//})

//m := mat.NewDense(1, 1, []float64{
//	2.0,
//})
//d := mat.NewDense(1,1,[]float64{
//	3.0,
//})
//x := &ng.Variable{Data:m}
//x.Data=x.Data.Grow(3,3).(*mat.Dense)
//x.Data.Scale(2, x.Data)
//x.Name="x"
//print("x:",dotVar(x,true))
//y1 := pow(x,2)
//y1.Name="y"
//print("y:",dotVar(y1,true))
//print("func:",dotFunc(y1.Creator))
//print(getDotGraph(y1,true))
//plotDotGraph(y1,true,"/Users/haihui/Downloads/pow.png")
////printDense("y",y.Data) //4

//check A numdiff
//A := NewFunction(&PowFunc{C:2})
//y := A.Run(x)
//y.Backward(true)
//printDense("xg",x.Grad.Data)
//f := A.Run
//g := numericalDiff(f,x,)
//printDense("dg",g)

////check b numdiff
//A := NewFunction(&expFunc{})
//y:=A.Run(x)
//y.Backward(false)
//printDense("xg", x.Grad.Data)
//f := A.Run
//g := numericalDiff(f,x)
//printDense("dg",g)

//com:= func (x...*ng.Variable) *ng.Variable {
//	a := mul(x[0],x[0])
//	b := exp(a)
//	y := mul(b,b)
//	return y
//}
//y := com(x)
//
//dy := numericalDiff(com,x)
//printDense("dg",dy)
//
//y.Backward(true)
//gx:=x.Grad
//printDense("gx",gx.Data)
//
//x.ClearGrade()
//gx.Backward(true)
//gx2:=x.Grad
//printDense("gx2",gx2.Data)
//plotDotGraph(gx2,true,"/Users/haihui/Downloads/pow.png")


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



