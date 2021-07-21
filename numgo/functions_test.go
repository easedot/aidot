package numgo

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSum (t *testing.T){
	var tests = []struct{
		x *Variable
		want *Variable
	}{
		{NewVec(1,2,3,4,5,6), NewVar(21)},
		{NewVar(21), NewVar(21)},
		{NewMat(2,3,1,2,3,4,5,6), NewVar(21)},
	}
	testFunc:= Sum
	for _,test :=range tests{
		got:=testFunc(test.x)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s",test.x.Sprint("x"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0:=test.x.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i)
		}, test.x)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
	}
}

func TestMax (t *testing.T){
	var tests = []struct{
		x *Variable
		axis interface{}
		want *Variable
	}{
		{NewMat(2,3,1,2,3,3,2,1),nil,NewVec(3)},
		{NewMat(2,3,1,2,3,3,2,1),0,NewVec(3,2,3)},
		{NewMat(2,3,1,2,3,3,2,1),1,NewMat(2,1,3,3)},
	}
	testFunc:= Max
	for _,test :=range tests{
		got:=testFunc(test.x,test.axis)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s",test.x.Sprint("x"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0:=test.x.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i,test.axis)
		}, test.x)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-6){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
	}
}

func TestMin (t *testing.T){
	var tests = []struct{
		x *Variable
		axis interface{}
		want *Variable
	}{
		{NewMat(2,3,1,2,3,3,2,1),0,NewVec(1,2,1)},
		{NewMat(2,3,1,2,3,3,2,1),1,NewMat(2,1,1,1)},
		{NewMat(2,3,1,2,3,3,2,1),nil,NewVec(1)},
	}
	testFunc:= Min
	for _,test :=range tests{
		got:=testFunc(test.x,test.axis)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0:=test.x.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i,test.axis)
		}, test.x)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-4){
			t.Errorf("\n%s\n%s\n%s\n%s",test.x.Sprint("x0"),test.x.Sprint("x1"),g0.Sprint("g0"),dc0.Sprint("wg0"))
		}
	}
}

func TestClip (t *testing.T){
	var tests = []struct{
		x *Variable
		max,min float64
		want *Variable
	}{
		{NewMat(2,3,1,2,3,3,2,1),2,1,NewMat(2,3,1,2,2,2,2,1)},
	}
	testFunc:= Clip
	for _,test :=range tests{
		got:=testFunc(test.x,test.min,test.max)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0:=test.x.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i,test.min,test.max)
		}, test.x)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-8){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),g0.Sprint("g"),dc0.Sprint("wg"))
		}
	}
}

func TestSoftmax(t *testing.T){
	var tests = []struct{
		x *Variable
		axis interface{}
		want *Variable
	}{
		{NewVec(0.3,2.9,4.0),1,NewVec(0.01821127,0.24519181,0.73659691)},
		{NewMat(2,3,0,1,2,0,2,4),1,NewMat(2,3,0.09003,0.24473,0.66524,0.01588,0.11731,0.86681)},
	}
	testFunc:= Softmax
	for _,test :=range tests{
		got:=testFunc(test.x,test.axis)
		if !mat.EqualApprox(got.Data,test.want.Data,1e-4){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0:=test.x.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i,test.axis)
		}, test.x)
		if mat.EqualApprox(g0.Data,dc0.Data,1e-8){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),g0.Sprint("g"),dc0.Sprint("wg"))
		}
	}
}
func TestSoftmaxCrossEntroy(t *testing.T){
	var tests = []struct{
		x *Variable
		t []int
		want *Variable
	}{
		{NewMat(2,4,-1,0,1,2,2,0,1,-1),[]int{3,0},NewMat(1,1,0.44018972)},
	}
	testFunc:= SoftmaxCrossEntroy
	for _,test :=range tests{
		got:=testFunc(test.x,test.t)
		if !mat.EqualApprox(got.Data,test.want.Data,1e-4){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0:=test.x.Grad
		dc0:=NumericalDiff(func(i *Variable) *Variable {
			return testFunc(i,test.t)
		}, test.x)
		if !mat.EqualApprox(g0.Data,dc0.Data,1e-8){
			t.Errorf("\n%s\n%s\n%s",test.x.Sprint("x"),g0.Sprint("g"),dc0.Sprint("wg"))
		}
	}
}