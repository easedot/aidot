package numgo

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMaxFunc(t *testing.T){
	var tests = []struct{
		x *Variable
		axis interface{}
		keepDims bool
		want *Variable
	}{
		{NewMat(2,3,1,2,3,3,2,1),nil,true,NewVec(3)},
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(3,2,3)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,3,3)},
	}
	testFunc := _max
	for _,test :=range tests {
		got := testFunc(test.x,test.axis,test.keepDims)
		if !mat.Equal(got.Data, test.want.Data) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestArgMaxFunc(t *testing.T){
	var tests = []struct{
		x *Variable
		axis interface{}
		keepDims bool
		want *Variable
	}{
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(1,0,0)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,2,0)},
	}
	testFunc := _agrMax
	for _,test :=range tests {
		got := testFunc(test.x,test.axis.(int),test.keepDims)
		if !mat.Equal(got.Data, test.want.Data) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}

func TestMinFunc(t *testing.T){
	var tests = []struct{
		x *Variable
		axis interface{}
		keepDims bool
		want *Variable
	}{
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(1,2,1)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,1,1)},
		{NewMat(2,3,1,2,3,3,2,1),nil,true,NewVec(1)},
	}
	testFunc := _min
	for _,test :=range tests {
		got := testFunc(test.x,test.axis,test.keepDims)
		if !mat.Equal(got.Data, test.want.Data) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestSumTo(t *testing.T){
	var tests = []struct{
		x *Variable
		axis interface{}
		keepDims bool
		want *Variable
	}{
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(4,4,4)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,6,6)},
		{NewMat(2,3,1,2,3,3,2,1),nil,true,NewVec(12)},
	}
	testFunc := _sum
	for _,test :=range tests {
		got := testFunc(test.x,test.axis,test.keepDims)
		if !mat.Equal(got.Data, test.want.Data) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestMaxBackwardShape(t *testing.T) {
	var tests = []struct{
		x *Variable
		axis interface{}
		want []int
	}{
		{NewMat(1,6,1,2,3,3,2,1),nil,[]int{1,1}},
		{NewMat(2,2,1,2,1,2),1,[]int{2,1}},
		{NewMat(2,2,1,2,1,2),[]int{0,1},[]int{1,1}},
		{NewMat(2,2,1,2,1,2),nil,[]int{1,1}},
		{NewMat(2,2,1,2,1,2),0,[]int{1,2}},
	}
	testFunc := _maxBackwardShape
	for _,test :=range tests {
		got := testFunc(test.x,test.axis)
		if !reflect.DeepEqual(got,test.want) {
			t.Errorf("\n%s\ny:%v\nw:%v", test.x.Sprint("x"), got, test.want)
		}
	}

}