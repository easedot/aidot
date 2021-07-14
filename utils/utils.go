package utils

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/plotter"
)
var NegFunc=func(_,_ int,v float64) float64{return -v}
var SinFunc=func(_,_ int,v float64) float64{return math.Sin(v)}
var CosFunc=func(_,_ int,v float64) float64{return math.Cos(v)}
var TanghFunc=func(_,_ int,v float64) float64{return math.Tanh(v)}
var SqrtFunc=func(_,_ int,v float64) float64{return math.Sqrt(v)}

//func Rand(){
//	rand.Float64()
//	rand.Perm(10)
//	//rand.Seed()
//}

func Rand(r,c int)[]float64{
	size:=r*c
	ra:=rand.New(rand.NewSource(time.Now().UnixNano()))
	s:=make([]float64,size,size)
	for i :=range s{
		s[i]=ra.Float64()
	}
	return s
}
func RandN(r,c int)[]float64{
	size:=r*c
	ra:=rand.New(rand.NewSource(time.Now().UnixNano()))
	s:=make([]float64,size,size)
	for i :=range s{
		s[i]=ra.NormFloat64()
	}
	return s
}

func RandE(r,c int)[]float64{
	size:=r*c
	ra:=rand.New(rand.NewSource(time.Now().UnixNano()))
	s:=make([]float64,size,size)
	for i :=range s{
		s[i]=ra.ExpFloat64()
	}
	return s
}

func Arange(start, stop, step float64) []float64 {
	N := int(math.Ceil((stop - start) / step));
	rnge := make([]float64, N, N)
	i := 0
	for x := start; x < stop; x += step {
		rnge[i] = x;
		i += 1
	}
	return rnge
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

func flatten(f [][]float64) (r, c int, d []float64) {
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



