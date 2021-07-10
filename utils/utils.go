package utils

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)
var NegFunc=func(_,_ int,v float64) float64{return -v}


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


func PrintDense(name string,x *mat.Dense) {
	fx := mat.Formatted(x, mat.Prefix("      "), mat.Squeeze())
	fmt.Printf("%3s = %v\n",name, fx)
}

func SprintDense(name string,x *mat.Dense) string {
	fx := mat.Formatted(x, mat.Prefix("       "), mat.Squeeze())
	rst := fmt.Sprintf("%3s = %v\n",name, fx)
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
