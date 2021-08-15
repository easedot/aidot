package tensor

import (
	"fmt"
	"testing"
)

func ExampleBroadcast() {
	x := NewArangeN(2, 3)
	y := NewVar(1)
	z := Add(x, y)
	z.Print("z")
	// Output:
	//z
	//[  1.000000   2.000000   3.000000 ]
	//[  4.000000   5.000000   6.000000 ]
}

func ExampleNewZero() {
	t := NewZero(3, 3)
	fmt.Printf("%v\n", t.Shape())
	fmt.Printf("%v\n", t.Strides)
	t.Print("")
	// Output:
	//[3 3]
	//[3 1]
	//[  0.000000   0.000000   0.000000 ]
	//[  0.000000   0.000000   0.000000 ]
	//[  0.000000   0.000000   0.000000 ]
}

func ExampleNewArangeN() {
	t := NewArangeN(3, 3)
	t.Print("")
	// Output:
	//[  0.000000   1.000000   2.000000 ]
	//[  3.000000   4.000000   5.000000 ]
	//[  6.000000   7.000000   8.000000 ]
}

func TestNew(t *testing.T) {
	//x3 := NewVec(1, 2, 3)
	//x3.Print("x3")

	//x := NewArangeN(2, 2)
	//x1 := NewArange(5, 8, 1).View(2, 2)
	//x1 := NewArangeN(2)
	//x.Print("x")
	//x1.Print("x1")
	//fmt.Printf("cdata:%v\n", x.data)
	//fmt.Printf("shape:%v\n", x.shape)
	//fmt.Printf("strid:%v\n", x.Strides)
	//y := x.SumTo(1, false)
	//y := x.MaxTo(0, false)
	//y := x.ArgMax(1, true)
	//y := x.MeanTo(1, true)
	//y := x.SliceSel(0, false, []int{0, 2}, []int{1, 2}).View(-1, 1)
	//y := x.SliceSelect(0, true, 0, 2)
	//y := x.Index(1, 1).Reshape(1, -1)
	//y := x.Slice(1, 2, 1).Reshape(1, -1)
	//y := Cat(1, x, x)
	//y := x.View(6, -1)
	//y := x.Clone()
	//y := Add(x, x1)
	//y := Dot(x, x)
	//y := Mul(x3, x3)
	//y := Dot(x, x)
	//y.Print("y:")
	//y = Sum(y)
	//y.Print("sum:")
	//fmt.Printf("cdata:%v\n", y.data)
	//fmt.Printf("shap:%v\n", y.shape)
	//fmt.Printf("strid:%v\n", y.Strides)
}

func TestAdd(t *testing.T) {
	var tests = []struct {
		x0, x1 *Tensor
		want   *Tensor
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewVec(2, 4, 6)},
		{NewVec(1, 2, 3), NewVar(10), NewVec(11, 12, 13)},
		{NewVar(10), NewVec(1, 2, 3), NewVec(11, 12, 13)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVec(1, 2, 3), NewData([]float64{2, 4, 6, 2, 4, 6}, 2, 3)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVar(10), NewData([]float64{11, 12, 13, 11, 12, 13}, 2, 3)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewData([]float64{1, 2, 3}, 3, 1), NewData([]float64{2, 3, 5, 3, 5, 6}, 3, 2)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewVar(10), NewData([]float64{11, 12, 13, 11, 12, 13}, 3, 2)},
	}
	testFunc := Add
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !Eq(got, test.want) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func BenchmarkAdd(b *testing.B) {
	x := NewVec(1, 2, 3)
	y := NewVec(1, 2, 3)
	for i := 0; i < b.N; i++ {
		Add(x, y)
	}
}
func benchmarkAdd(b *testing.B, n int) {
	x := NewRand(n, n)
	y := NewRand(n, n)
	for i := 0; i < b.N; i++ {
		Add(x, y)
	}
}
func BenchmarkAdd100(b *testing.B)   { benchmarkAdd(b, 100) }
func BenchmarkAdd1000(b *testing.B)  { benchmarkAdd(b, 1000) }
func BenchmarkAdd10000(b *testing.B) { benchmarkAdd(b, 10000) }

func ExampleAdd() {
	x := NewVec(1, 2, 3)
	y := NewVec(4, 5, 6)
	z := Add(x, y)
	z.Print("")
	// Output:
	// 5.000000   7.000000   9.000000
}
func TestSub(t *testing.T) {
	var tests = []struct {
		x0, x1 *Tensor
		want   *Tensor
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewData([]float64{0, 0, 0}, 3)},
		{NewVec(1, 2, 3), NewVar(1), NewVec(0, 1, 2)},
		{NewVar(10), NewVec(1, 2, 3), NewVec(9, 8, 7)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVec(1, 2, 3), NewData([]float64{0, 0, 0, 0, 0, 0}, 2, 3)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVar(1), NewData([]float64{0, 1, 2, 0, 1, 2}, 2, 3)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewData([]float64{1, 2, 3}, 3, 1), NewData([]float64{0, 1, 1, -1, -1, 0}, 3, 2)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewVar(1), NewData([]float64{0, 1, 2, 0, 1, 2}, 3, 2)},
	}
	testFunc := Sub
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !Eq(got, test.want) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestMul(t *testing.T) {
	var tests = []struct {
		x0, x1 *Tensor
		want   *Tensor
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewData([]float64{1, 4, 9}, 3)},
		{NewVec(1, 2, 3), NewVar(10), NewVec(10, 20, 30)},
		{NewVar(10), NewVec(1, 2, 3), NewVec(10, 20, 30)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVec(1, 2, 3), NewData([]float64{1, 4, 9, 1, 4, 9}, 2, 3)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVar(10), NewData([]float64{10, 20, 30, 10, 20, 30}, 2, 3)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewData([]float64{1, 2, 3}, 3, 1), NewData([]float64{1, 2, 6, 2, 6, 9}, 3, 2)},
		{NewData([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewVar(10), NewData([]float64{10, 20, 30, 10, 20, 30}, 3, 2)},
	}
	testFunc := Mul
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !Eq(got, test.want) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestDiv(t *testing.T) {
	var tests = []struct {
		x0, x1 *Tensor
		want   *Tensor
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewData([]float64{1, 1, 1}, 3)},
		{NewVec(1, 2, 3), NewVar(1), NewVec(1, 2, 3)},
		{NewVar(10), NewVec(1, 2, 5), NewVec(10, 5, 2)},
		{NewData([]float64{2, 4, 6, 2, 4, 6}, 2, 3), NewVec(1, 2, 3), NewData([]float64{2, 2, 2, 2, 2, 2}, 2, 3)},
		{NewData([]float64{2, 4, 6, 2, 4, 6}, 2, 3), NewVar(2), NewData([]float64{1, 2, 3, 1, 2, 3}, 2, 3)},
		{NewData([]float64{2, 4, 6, 2, 4, 6}, 3, 2), NewData([]float64{1, 2, 2}, 3, 1), NewData([]float64{2, 4, 3, 1, 2, 3}, 3, 2)},
		{NewData([]float64{2, 4, 6, 2, 4, 6}, 3, 2), NewVar(2), NewData([]float64{1, 2, 3, 1, 2, 3}, 3, 2)},
	}
	testFunc := Div
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !Eq(got, test.want) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestDot(t *testing.T) {
	var tests = []struct {
		x0, x1 *Tensor
		want   *Tensor
	}{
		{NewData([]float64{1, 2, 3, 4, 5, 6}, 2, 3), NewData([]float64{1, 2, 3, 4, 5, 6}, 2, 3).T(), NewData([]float64{14, 32, 32, 77}, 2, 2)},
		{NewData([]float64{1, 2, 3, 4}, 2, 2), NewData([]float64{5, 6, 7, 8}, 2, 2), NewData([]float64{19, 22, 43, 50}, 2, 2)},
		{NewVec(1, 2, 3), NewVec(4, 5, 6), NewVar(32)},
	}
	testFunc := Dot
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !Eq(got, test.want) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestMask(t *testing.T) {
	var tests = []struct {
		x0   *Tensor
		cond func(pos []int, v float64) float64
		want *Tensor
	}{
		{
			NewVec(1, 2, 3, 4, 5, 6).View(2, 3),
			func(pos []int, v float64) float64 {
				ma := NewVec(1, 2, 2, 4, 4, 6).View(2, 3)
				if ma.Get(pos...) == v {
					return 1
				} else {
					return 0
				}
			},
			NewVec(1, 1, 0, 1, 0, 1).View(2, 3),
		},
	}
	for _, test := range tests {
		got := test.x0.Clone()
		got.ApplyPos(test.cond)
		if !Eq(got, test.want) {
			t.Errorf("\n %s %s %s", test.x0.Sprint("x0"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestMax(t *testing.T) {
	var tests = []struct {
		x        *Tensor
		axis     int
		keepDims bool
		want     *Tensor
	}{
		//{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), nil, true, NewVec(3)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, false, NewVec(3, 2, 3)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, true, NewData([]float64{3, 3}, 2, 1)},
	}
	for _, test := range tests {
		got := test.x.MaxTo(test.axis, test.keepDims)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestArgMax(t *testing.T) {
	var tests = []struct {
		x        *Tensor
		axis     int
		keepDims bool
		want     *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, false, NewVec(1, 0, 0)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, true, NewData([]float64{2, 0}, 2, 1)},
	}
	for _, test := range tests {
		got := test.x.ArgMax(test.axis, test.keepDims)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestMin(t *testing.T) {
	var tests = []struct {
		x        *Tensor
		axis     int
		keepDims bool
		want     *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, false, NewVec(1, 2, 1)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, true, NewData([]float64{1, 1}, 2, 1)},
		//{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), nil, true, NewVec(1)},
	}
	for _, test := range tests {
		got := test.x.MinTo(test.axis, test.keepDims)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestSum(t *testing.T) {
	var tests = []struct {
		x        *Tensor
		axis     int
		keepDims bool
		want     *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, false, NewVec(4, 4, 4)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, false, NewData([]float64{6, 6}, 2)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, true, NewVec(4, 4, 4).View(1, 3)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, true, NewData([]float64{6, 6}, 2, 1)},
		//{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), nil, true, NewVec(12)},
	}
	for _, test := range tests {
		got := test.x.SumTo(test.axis, test.keepDims)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestMean(t *testing.T) {
	var tests = []struct {
		x       *Tensor
		dim     int
		keepDim bool
		want    *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, false, NewVec(2, 2, 2)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, false, NewVec(2, 2)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, true, NewVec(2, 2, 2).View(1, -1)},
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, true, NewVec(2, 2).View(2, -1)},
	}
	for _, test := range tests {
		got := test.x.Clone().MeanTo(test.dim, test.keepDim)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestNeg(t *testing.T) {
	var tests = []struct {
		x    *Tensor
		want *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), NewData([]float64{-1, -2, -3, -3, -2, -1}, 2, 3)},
	}
	for _, test := range tests {
		got := Neg(test.x)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestPow(t *testing.T) {
	var tests = []struct {
		x    *Tensor
		want *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), NewData([]float64{1, 4, 9, 9, 4, 1}, 2, 3)},
	}
	for _, test := range tests {
		got := Pow(test.x, 2)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestClip(t *testing.T) {
	var tests = []struct {
		x        *Tensor
		min, max float64
		want     *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, 2, NewData([]float64{1, 2, 2, 2, 2, 1}, 2, 3)},
	}
	for _, test := range tests {
		got := Clip(test.x, test.min, test.max)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestRows(t *testing.T) {
	var tests = []struct {
		x    *Tensor
		s    []int
		want *Tensor
	}{
		{NewData([]float64{1, 2, 3, 3, 2, 1}, 2, 3), []int{1}, NewVec(3, 2, 1).View(1, -1)},
		{NewData([]float64{1, 2, 3, 3, 2, 1, 4, 5, 6}, 3, 3), []int{0, 2}, NewData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)},
		{NewData([]float64{1, 2, 3, 3, 2, 1, 4, 5, 6, 6, 5, 4}, 4, 3), []int{1, 3}, NewData([]float64{3, 2, 1, 6, 5, 4}, 2, 3)},
	}
	for _, test := range tests {
		got := test.x.Slices(0, test.s...)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestRowsCol(t *testing.T) {
	var tests = []struct {
		x       *Tensor
		dim     int
		keepDim bool
		r, c    []int
		want    *Tensor
	}{
		{NewArangeN(3, 4), 0, false, []int{0, 2}, []int{1, 3}, NewVec(1, 11)},
		//{NewArangeN(3, 4, 4), 0, false, []int{0, 1}, []int{0, 2}, NewData([]float64{2, 1, 4}, 3, 1)},
	}
	for _, test := range tests {
		got := test.x.SliceSel(test.dim, test.keepDim, test.r, test.c)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestView(t *testing.T) {
	var tests = []struct {
		x     *Tensor
		shape []int
		want  *Tensor
	}{
		{NewArangeN(3, 2), []int{2, 3}, NewArangeN(2, 3)},
		{NewArangeN(3, 2), []int{-1, 6}, NewArangeN(1, 6)},
		{NewArangeN(3, 2), []int{-1, 3}, NewArangeN(2, 3)},
		{NewArangeN(3, 2), []int{6, -1}, NewArangeN(6, 1)},
		{NewArangeN(3, 2), []int{2, -1}, NewArangeN(2, 3)},
	}
	for _, test := range tests {
		got := test.x.View(test.shape...)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("want"))
		}
	}
}

func TestIndex(t *testing.T) {
	var tests = []struct {
		x        *Tensor
		idx, dim int
		want     *Tensor
	}{
		{NewArangeN(3, 2), 1, 0, NewData([]float64{2, 3}, 2)},
		{NewArangeN(3, 2), 1, 1, NewData([]float64{1, 3, 5}, 3)},
	}
	for _, test := range tests {
		got := test.x.Index(test.idx, test.dim)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("want"))
		}
	}
}

func TestSlice(t *testing.T) {
	var tests = []struct {
		x               *Tensor
		start, end, dim int
		want            *Tensor
	}{
		{NewArangeN(3, 2), 1, 2, 0, NewData([]float64{2, 3}, 1, 2)},
		{NewArangeN(3, 2), 1, 2, 1, NewData([]float64{1, 3, 5}, 3, 1)},
	}
	for _, test := range tests {
		got := test.x.Slice(test.start, test.end, test.dim)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("want"))
		}
	}
}

func TestCat(t *testing.T) {
	var tests = []struct {
		x    *Tensor
		dim  int
		want *Tensor
	}{
		{NewArangeN(3, 2), 0, NewData([]float64{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}, 6, 2)},
		{NewArangeN(3, 2), 1, NewData([]float64{0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5}, 3, 4)},
	}
	for _, test := range tests {
		got := Cat(test.dim, test.x, test.x)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("want"))
		}
	}
}

func TestSqueeze(t *testing.T) {
	var tests = []struct {
		x    *Tensor
		want []int
	}{
		{NewZero(2, 1, 2, 1, 2), []int{2, 2, 2}},
	}
	for _, test := range tests {
		got := Squeeze(test.x)
		if !testEqInt(got.shape, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), fmt.Sprintf("want:%v", test.want))
		}
	}
}
func TestUnSqueeze(t *testing.T) {
	var tests = []struct {
		x    *Tensor
		dim  int
		want *Tensor
	}{
		{NewVec(1, 2, 3, 4), 0, NewVec(1, 2, 3, 4).View(1, 4)},
		{NewVec(1, 2, 3, 4), 1, NewVec(1, 2, 3, 4).View(4, 1)},
	}
	for _, test := range tests {
		got := UnSqueeze(test.x, test.dim)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestSaveLoad(t *testing.T) {
	var tests = []struct {
		x    *Tensor
		name string
	}{
		{NewVec(1, 2, 3, 4, 5), "test.gob"},
	}
	for _, test := range tests {
		test.x.Save(test.name)
		y := NewVec()
		y.Load(test.name)
		if Equal(test.x, y) {
			t.Errorf("\n%s\n%s", test.x.Sprint("save"), y.Sprint("load"))
		}
	}
}
