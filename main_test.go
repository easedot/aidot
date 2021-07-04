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
			panic("bad test: ragged input")
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
	if got:=square(x);!mat.Equal(got.Data,r){
		t.Errorf("Square error\n x:\n%s y:\n%s ",sprintDense(x.Data),sprintDense(got.Data))
	}
}
func TestBackward(t *testing.T){
	m := mat.NewDense(2, 2, []float64{
		2.0,2.0,
		2.0,2.0,
	})
	x:=&Variable{Data: m}
	f:=NewFunction(&Square{})
	nd :=numericalDiff(f.Run,x,0)
	y:=square(x)
	y.Backward()
	if !mat.EqualApprox(y.Grad, nd,1e-4){
		t.Errorf("Square error\n x:\n%s y:\n%s ",sprintDense(x.Data),sprintDense(y.Data))
	}
}