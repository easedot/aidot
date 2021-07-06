package main

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)
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



func TestForward(t *testing.T){
	//tt:=[][]float64{
	//	{1,2,3,4},
	//	{2,3,4,5},
	//}
	//println(len(tt[len(tt)-1]))
	m := mat.NewDense(2, 2, []float64{
		4.0,4.0,
		4.0,4.0,
	})

	r := mat.NewDense(2, 2, []float64{
		32.0,32.0,
		32.0,32.0,
	})
	x:=&Variable{Data: m}
	if got:=pow(x,2);!mat.Equal(got.Data,r){
		t.Errorf("Square error\n x:\n%s y:\n%s ",sprintDense(x.Data),sprintDense(got.Data))
	}
}
func TestBackward(t *testing.T){
	//m := mat.NewDense(1, 1, []float64{
	//	2.0,
	//})
	m := mat.NewDense(2, 2, []float64{
		2.0,2.0,
		2.0,2.0,
	})
	f:= func (x...*Variable) *Variable {
		y := mul(x[0],x[0])
		return y
	}
	x:=&Variable{Data: m}
	//f:=NewFunction(&Pow{C:2})
	nd :=numericalDiff(f,x)
	printDense("dg",nd)
	y:=f(x)
	printDense("yd",y.Data)
	y.Backward(true)
	printDense("xg",x.Grad.Data)
	if !mat.EqualApprox(x.Grad.Data, nd,1e-4){
		t.Errorf("Square error\n xg:\n%s dg:\n%s ",sprintDense(x.Grad.Data),sprintDense(nd))
	}
}