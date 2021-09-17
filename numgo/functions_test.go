package numgo

import (
	"testing"

	nt "test_ai/tensor"
)

func TestSum(t *testing.T) {
	var tests = []struct {
		x        *Variable
		keepDims bool
		want     *Variable
	}{
		{NewVec(1, 2, 3, 4, 5, 6), false, NewVar(21)},
		{NewVar(21), false, NewVar(21)},
		{NewMat([]float64{1, 2, 3, 4, 5, 6}, 2, 3), false, NewVar(21)},
	}
	testFunc := Sum
	for index, test := range tests {
		got := testFunc(test.x, test.keepDims)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0 := test.x.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, false)
		}, test.x)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
	}
}

func TestMax(t *testing.T) {
	var tests = []struct {
		x        *Variable
		axis     int
		keepDims bool
		want     *Variable
	}{
		//{NewMat([]float64{}, 2, 3, 1, 2, 3, 3, 2, 1), nil, NewVec(3)},
		{NewMat([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, false, NewVec(3, 2, 3)},
		{NewMat([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, false, NewMat([]float64{3, 3}, 2, 1)},
	}
	testFunc := Max
	for index, test := range tests {
		got := testFunc(test.x, test.keepDims, test.axis)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0 := test.x.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.keepDims, test.axis)
		}, test.x)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
	}
}

func TestMin(t *testing.T) {
	var tests = []struct {
		x        *Variable
		axis     int
		keepDims bool
		want     *Variable
	}{
		{NewMat([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 0, false, NewVec(1, 2, 1)},
		{NewMat([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 1, false, NewMat([]float64{1, 1}, 2, 1)},
		//{NewMat([]float64{}, 2, 3, 1, 2, 3, 3, 2, 1), nil, NewVec(1)},
	}
	testFunc := Min
	for index, test := range tests {
		got := testFunc(test.x, test.keepDims, test.axis)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0 := test.x.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.keepDims, test.axis)
		}, test.x)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s%s", index, test.x.Sprint("x0"), test.x.Sprint("x1"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
	}
}

func TestClip(t *testing.T) {
	var tests = []struct {
		x        *Variable
		max, min float64
		want     *Variable
	}{
		{NewMat([]float64{1, 2, 3, 3, 2, 1}, 2, 3), 2, 1, NewMat([]float64{1, 2, 2, 2, 2, 1}, 2, 3)},
	}
	testFunc := Clip
	for index, test := range tests {
		got := testFunc(test.x, test.min, test.max)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0 := test.x.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.min, test.max)
		}, test.x)
		//1e-8
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), g0.Sprint("g"), dc0.Sprint("wg"))
		}
	}
}

func TestSoftmax(t *testing.T) {
	var tests = []struct {
		x    *Variable
		axis int
		want *Variable
	}{
		{NewVec(0.3, 2.9, 4.0), 1, NewVec(0.01821127, 0.24519181, 0.73659691)},
		{NewMat([]float64{0, 1, 2, 0, 2, 4}, 2, 3), 1, NewMat([]float64{0.09003, 0.24473, 0.66524, 0.01588, 0.11731, 0.86681}, 2, 3)},
	}
	testFunc := Softmax
	for index, test := range tests {
		got := testFunc(test.x, test.axis)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0 := test.x.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.axis)
		}, test.x)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), g0.Sprint("g"), dc0.Sprint("wg"))
		}
	}
}
func TestSoftmaxCrossEntroy(t *testing.T) {
	var tests = []struct {
		x    *Variable
		t    []float64
		want *Variable
	}{
		{NewMat([]float64{-1, 0, 1, 2, 2, 0, 1, -1}, 2, 4), []float64{3, 0}, NewMat([]float64{0.44018972}, 1, 1)},
	}
	testFunc := SoftmaxCrossEntroy
	for index, test := range tests {
		got := testFunc(test.x, test.t)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("input"), got.Sprint("got"), test.want.Sprint("want"))
		}
		got.Backward(true)
		g0 := test.x.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.t)
		}, test.x)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("input"), g0.Sprint("got"), dc0.Sprint("want"))
		}
	}
}

func TestSigmoid(t *testing.T) {
	var tests = []struct {
		x    *Variable
		want *Variable
	}{
		{NewMat([]float64{0, 1, 2, 0, 2, 4}, 2, 3), NewMat([]float64{0.50000, 0.73106, 0.88080, 0.50000, 0.88080, 0.98201}, 2, 3)},
	}
	testFunc := Sigmoid
	for index, test := range tests {
		got := testFunc(test.x)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0 := test.x.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i)
		}, test.x)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s", index, test.x.Sprint("x"), g0.Sprint("g"), dc0.Sprint("wg"))
		}
	}
}

func TestLinear(t *testing.T) {
	var tests = []struct {
		x, W, b *Variable
		want    *Variable
	}{
		{NewMat([]float64{1, 2, 3, 4, 5, 6}, 2, 3), NewVariable(nt.NewVec(1, 2, 3, 4, 5, 6).View(2, 3).T()), NewVar(0), NewMat([]float64{14, 32, 32, 77}, 2, 2)},
		{NewMat([]float64{1, 2, 3, 4}, 2, 2), NewMat([]float64{5, 6, 7, 8}, 2, 2), NewVar(0), NewMat([]float64{19, 22, 43, 50}, 2, 2)},
	}
	testFunc := Linear
	for index, test := range tests {
		got := testFunc(test.x, test.W, test.b)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\nID:%d \n %s %s %s %s", index, test.x.Sprint("x"), test.W.Sprint("w"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0, g1 := test.x.Grad, test.W.Grad

		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.W, test.b)
		}, test.x)
		dc1 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x, i, test.b)
		}, test.W)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\nID:%d \n%s%s%s%s", index, test.x.Sprint("x"), test.W.Sprint("w"), g0.Sprint("gx"), dc0.Sprint("wgx"))
		}
		if !nt.DeepEqual(g1.Data, dc1.Data) {
			t.Errorf("\nID:%d \n%s%s%s%s", index, test.x.Sprint("x"), test.W.Sprint("w"), g1.Sprint("gw"), dc1.Sprint("wgw"))
		}
	}
}
