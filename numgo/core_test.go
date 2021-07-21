package numgo

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAdd (t *testing.T){
	var tests = []struct{
		x0,x1 *Variable
		want *Variable
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,2,4,6)},
		{NewVec(1,2,3),NewVar(10), NewVec(11,12,13)},
		{NewVar(10),NewVec(1,2,3), NewVec(11,12,13)},
		{NewMat(2,3,1,2,3,1,2,3),NewVec(1,2,3), NewMat(2,3,2,4,6,2,4,6)},
		{NewMat(2,3,1,2,3,1,2,3),NewVar(10), NewMat(2,3,11,12,13,11,12,13)},
		{NewMat(3,2,1,2,3,1,2,3),NewMat(3,1,1,2,3), NewMat(3,2,2,3,5,3,5,6)},
		{NewMat(3,2,1,2,3,1,2,3),NewVar(10), NewMat(3,2,11,12,13,11,12,13)},
	}
	testFunc:= Add
	for _,test :=range tests{
		got:=testFunc(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0,g1:=test.x0.Grad,test.x1.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)

		dc1:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0, i)
		}, test.x1)

		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
		if !mat.EqualApprox(g1.Data,dc1.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g1.Sprint("g1"),dc1.Sprint("wg1"))
		}

	}
}
func TestSub (t *testing.T){
	var tests = []struct{
		x0,x1 *Variable
		want *Variable
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,0,0,0)},
		{NewVec(1,2,3),NewVar(1), NewVec(0,1,2)},
		{NewVar(10),NewVec(1,2,3), NewVec(9,8,7)},
		{NewMat(2,3,1,2,3,1,2,3),NewVec(1,2,3), NewMat(2,3,0,0,0,0,0,0)},
		{NewMat(2,3,1,2,3,1,2,3),NewVar(1), NewMat(2,3,0,1,2,0,1,2)},
		{NewMat(3,2,1,2,3,1,2,3),NewMat(3,1,1,2,3), NewMat(3,2,0,1,1,-1,-1,0)},
		{NewMat(3,2,1,2,3,1,2,3),NewVar(1), NewMat(3,2,0,1,2,0,1,2)},
	}
	testFunc := Sub
	for _,test :=range tests{
		got:= testFunc(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)

		g0,g1:=test.x0.Grad,test.x1.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)

		dc1:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0, i)
		}, test.x1)

		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
		if !mat.EqualApprox(g1.Data,dc1.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g1.Sprint("g1"),dc1.Sprint("wg1"))
		}

	}
}
func TestMul(t *testing.T){
	var tests = []struct{
		x0,x1 *Variable
		want *Variable
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,1,4,9)},
		{NewVec(1,2,3),NewVar(10), NewVec(10,20,30)},
		{NewVar(10),NewVec(1,2,3), NewVec(10,20,30)},
		{NewMat(2,3,1,2,3,1,2,3),NewVec(1,2,3), NewMat(2,3,1,4,9,1,4,9)},
		{NewMat(2,3,1,2,3,1,2,3),NewVar(10), NewMat(2,3,10,20,30,10,20,30)},
		{NewMat(3,2,1,2,3,1,2,3),NewMat(3,1,1,2,3), NewMat(3,2,1,2,6,2,6,9)},
		{NewMat(3,2,1,2,3,1,2,3),NewVar(10), NewMat(3,2,10,20,30,10,20,30)},
	}
	testFunc := Mul
	for _,test :=range tests{
		got:= testFunc(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0,g1:=test.x0.Grad,test.x1.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)
		dc1:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0,i)
		}, test.x1)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
		if !mat.EqualApprox(g1.Data,dc1.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g1.Sprint("g1"),dc1.Sprint("wg1"))
		}

	}

}
func TestDiv(t *testing.T){
	var tests = []struct{
		x0,x1 *Variable
		want *Variable
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,1,1,1)},
		{NewVec(1,2,3),NewVar(1), NewVec(1,2,3)},
		{NewVar(10),NewVec(1,2,5), NewVec(10,5,2)},
		{NewMat(2,3,2,4,6,2,4,6),NewVec(1,2,3), NewMat(2,3,2,2,2,2,2,2)},
		{NewMat(2,3,2,4,6,2,4,6),NewVar(2), NewMat(2,3,1,2,3,1,2,3)},
		{NewMat(3,2,2,4,6,2,4,6),NewMat(3,1,1,2,2), NewMat(3,2,2,4,3,1,2,3)},
		{NewMat(3,2,2,4,6,2,4,6),NewVar(2), NewMat(3,2,1,2,3,1,2,3)},
	}
	for _,test :=range tests{
		testFunc := Div
		got:= testFunc(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0,g1:=test.x0.Grad,test.x1.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)
		dc1:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0,i)
		}, test.x1)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
		if !mat.EqualApprox(g1.Data,dc1.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g1.Sprint("g1"),dc1.Sprint("wg1"))
		}

	}

}
func TestMatmul(t *testing.T){
	var tests = []struct{
		x0,x1 *Variable
		want *Variable
	}{
		{NewMat(2,3,1,2,3,4,5,6),NewMat(2,3,1,2,3,4,5,6).T(), NewMat(2,2,14,32,32,77)},
		{NewMat(2,2,1,2,3,4),NewMat(2,2,5,6,7,8), NewMat(2,2,19,22,43,50)},
		{NewVec(1,2,3),NewVec(4,5,6).T(), NewVar(32)},

	}
	testFunc:=Matmul
	for _,test :=range tests{
		got:=testFunc(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0,g1:=test.x0.Grad,test.x1.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.x1)
		}, test.x0)
		dc1:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x0,i)
		}, test.x1)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
		if !mat.EqualApprox(g1.Data,dc1.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g1.Sprint("g1"),dc1.Sprint("wg1"))
		}
	}

}

func TestLinear(t *testing.T){
	var tests = []struct{
		x,W,b *Variable
		want *Variable
	}{
		{NewMat(2,3,1,2,3,4,5,6),NewMat(2,3,1,2,3,4,5,6).T(),NewVar(0), NewMat(2,2,14,32,32,77)},
		{NewMat(2,2,1,2,3,4),NewMat(2,2,5,6,7,8),NewVar(0), NewMat(2,2,19,22,43,50)},
		{NewVec(1,2,3),NewVec(4,5,6).T(),NewVar(0), NewVar(32)},
	}
	testFunc:= Linear
	for _,test :=range tests{
		got:=testFunc(test.x,test.W,test.b)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x.Sprint("x0"),test.W.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0,g1:=test.x.Grad,test.W.Grad

		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i, test.W,test.b)
		}, test.x)
		dc1:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(test.x,i,test.b)
		}, test.W)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x.Sprint("x0"),test.W.Sprint("x1"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
		if !mat.EqualApprox(g1.Data,dc1.Data,1e-4){
			t.Errorf("\n %s %s %s %s",test.x.Sprint("x0"),test.W.Sprint("x1"),g1.Sprint("g1"),dc1.Sprint("wg1"))
		}
	}

}

