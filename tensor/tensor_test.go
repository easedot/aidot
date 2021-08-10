package tensor

import (
	"fmt"
	"testing"
)

func TestNew(t *testing.T) {
	x := NewArangeN(3, 2)
	x.Print("x")
	fmt.Printf("cdata:%v\n", x.data)
	fmt.Printf("shape:%v\n", x.shape)
	fmt.Printf("strid:%v\n", x.Strides)
	y := x.Index(1, 1)
	//y := x.Slice(1, 2, 1)
	//y := Cat(1, x, x)
	y.Print("index:")

	fmt.Printf("cdata:%v\n", y.data)
	fmt.Printf("shap:%v\n", y.shape)
	fmt.Printf("strid:%v\n", y.Strides)
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
		got := test.x.Squeeze()
		got.Print("test")
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
		got := test.x.UnSqueeze(test.dim)
		if !Eq(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestBroadcast(t *testing.T) {
	x := NewZero(5, 2, 4, 1)
	y := NewZero(2, 1, 1)
	fmt.Printf("broadcastable:%t", isBoradcastable(x.shape, y.shape))
}

func ExampleNewTensor() {
	t := NewZero(3, 3)
	fmt.Printf("%v\n", t.Shape())
	fmt.Printf("%v\n", t.Strides)
	// Output:
	//[3 3]
	//[3 1]
}
