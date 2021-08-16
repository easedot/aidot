package numgo

import (
	"testing"

	nt "test_ai/tensor"
)

func TestAdd(t *testing.T) {
	var tests = []struct {
		x0, x1 *Variable
		want   *Variable
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewMat([]float64{2, 4, 6}, 1, 3)},
		{NewVec(1, 2, 3), NewVar(10), NewVec(11, 12, 13)},
		{NewVar(10), NewVec(1, 2, 3), NewVec(11, 12, 13)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVec(1, 2, 3), NewMat([]float64{2, 4, 6, 2, 4, 6}, 2, 3)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVar(10), NewMat([]float64{11, 12, 13, 11, 12, 13}, 2, 3)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewMat([]float64{1, 2, 3}, 3, 1), NewMat([]float64{2, 3, 5, 3, 5, 6}, 3, 2)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewVar(10), NewMat([]float64{11, 12, 13, 11, 12, 13}, 3, 2)},
	}
	testFunc := Add
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0, g1 := test.x0.Grad, test.x1.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)

		dc1 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0, i)
		}, test.x1)

		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
		if !nt.DeepEqual(g1.Data, dc1.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g1.Sprint("g1"), dc1.Sprint("wg1"))
		}

	}
}
func TestSub(t *testing.T) {
	var tests = []struct {
		x0, x1 *Variable
		want   *Variable
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewMat([]float64{0, 0, 0}, 1, 3)},
		{NewVec(1, 2, 3), NewVar(1), NewVec(0, 1, 2)},
		{NewVar(10), NewVec(1, 2, 3), NewVec(9, 8, 7)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVec(1, 2, 3), NewMat([]float64{0, 0, 0, 0, 0, 0}, 2, 3)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVar(1), NewMat([]float64{0, 1, 2, 0, 1, 2}, 2, 3)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewMat([]float64{1, 2, 3}, 3, 1), NewMat([]float64{0, 1, 1, -1, -1, 0}, 3, 2)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewVar(1), NewMat([]float64{0, 1, 2, 0, 1, 2}, 3, 2)},
	}
	testFunc := Sub
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)

		g0, g1 := test.x0.Grad, test.x1.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)

		dc1 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0, i)
		}, test.x1)

		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
		if !nt.DeepEqual(g1.Data, dc1.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g1.Sprint("g1"), dc1.Sprint("wg1"))
		}

	}
}
func TestMul(t *testing.T) {
	var tests = []struct {
		x0, x1 *Variable
		want   *Variable
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewMat([]float64{1, 4, 9}, 1, 3)},
		{NewVec(1, 2, 3), NewVar(10), NewVec(10, 20, 30)},
		{NewVar(10), NewVec(1, 2, 3), NewVec(10, 20, 30)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVec(1, 2, 3), NewMat([]float64{1, 4, 9, 1, 4, 9}, 2, 3)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 2, 3), NewVar(10), NewMat([]float64{10, 20, 30, 10, 20, 30}, 2, 3)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewMat([]float64{1, 2, 3}, 3, 1), NewMat([]float64{1, 2, 6, 2, 6, 9}, 3, 2)},
		{NewMat([]float64{1, 2, 3, 1, 2, 3}, 3, 2), NewVar(10), NewMat([]float64{10, 20, 30, 10, 20, 30}, 3, 2)},
	}
	testFunc := Mul
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0, g1 := test.x0.Grad, test.x1.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)
		dc1 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0, i)
		}, test.x1)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
		if !nt.DeepEqual(g1.Data, dc1.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g1.Sprint("g1"), dc1.Sprint("wg1"))
		}

	}

}
func TestDiv(t *testing.T) {
	var tests = []struct {
		x0, x1 *Variable
		want   *Variable
	}{
		{NewVec(1, 2, 3), NewVec(1, 2, 3), NewMat([]float64{1, 1, 1}, 1, 3)},
		{NewVec(1, 2, 3), NewVar(1), NewVec(1, 2, 3)},
		{NewVar(10), NewVec(1, 2, 5), NewVec(10, 5, 2)},
		{NewMat([]float64{2, 4, 6, 2, 4, 6}, 2, 3), NewVec(1, 2, 3), NewMat([]float64{2, 2, 2, 2, 2, 2}, 2, 3)},
		{NewMat([]float64{2, 4, 6, 2, 4, 6}, 2, 3), NewVar(2), NewMat([]float64{1, 2, 3, 1, 2, 3}, 2, 3)},
		{NewMat([]float64{2, 4, 6, 2, 4, 6}, 3, 2), NewMat([]float64{1, 2, 2}, 3, 1), NewMat([]float64{2, 4, 3, 1, 2, 3}, 3, 2)},
		{NewMat([]float64{2, 4, 6, 2, 4, 6}, 3, 2), NewVar(2), NewMat([]float64{1, 2, 3, 1, 2, 3}, 3, 2)},
	}
	for _, test := range tests {
		testFunc := Div
		got := testFunc(test.x0, test.x1)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0, g1 := test.x0.Grad, test.x1.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)
		dc1 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0, i)
		}, test.x1)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
		if !nt.DeepEqual(g1.Data, dc1.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g1.Sprint("g1"), dc1.Sprint("wg1"))
		}

	}

}
func TestMatmul(t *testing.T) {
	var tests = []struct {
		x0, x1 *Variable
		want   *Variable
	}{
		{NewMat([]float64{1, 2, 3, 4, 5, 6}, 2, 3), NewVariable(nt.NewVec(1, 2, 3, 4, 5, 6).View(2, 3).T()), NewMat([]float64{14, 32, 32, 77}, 2, 2)},
		{NewMat([]float64{1, 2, 3, 4}, 2, 2), NewMat([]float64{5, 6, 7, 8}, 2, 2), NewMat([]float64{19, 22, 43, 50}, 2, 2)},
		{NewVec(1, 2, 3), NewVariable(nt.NewVec(4, 5, 6).T()), NewVar(32)},
	}
	testFunc := Matmul
	for _, test := range tests {
		got := testFunc(test.x0, test.x1)
		if nt.Equal(got.Data, test.want.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
		}
		got.Backward(true)
		g0, g1 := test.x0.Grad, test.x1.Grad
		dc0 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)
		dc1 := NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0, i)
		}, test.x1)
		if !nt.DeepEqual(g0.Data, dc0.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
		if !nt.DeepEqual(g1.Data, dc1.Data) {
			t.Errorf("\n %s %s %s %s", test.x0.Sprint("x0"), test.x1.Sprint("x1"), g1.Sprint("g1"), dc1.Sprint("wg1"))
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
		{NewVec(1, 2, 3), NewVariable(nt.NewVec(4, 5, 6).T()), NewVar(0), NewVar(32)},
	}
	testFunc := Linear
	for _, test := range tests {
		got := testFunc(test.x, test.W, test.b)
		if !nt.DeepEqual(got.Data, test.want.Data) {
			t.Errorf("\n %s %s %s %s", test.x.Sprint("x0"), test.W.Sprint("x1"), got.Sprint("y"), test.want.Sprint("w"))
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
			t.Errorf("\n %s %s %s %s", test.x.Sprint("x0"), test.W.Sprint("x1"), g0.Sprint("g0"), dc0.Sprint("wg0"))
		}
		if !nt.DeepEqual(g1.Data, dc1.Data) {
			t.Errorf("\n %s %s %s %s", test.x.Sprint("x0"), test.W.Sprint("x1"), g1.Sprint("g1"), dc1.Sprint("wg1"))
		}
	}
}
