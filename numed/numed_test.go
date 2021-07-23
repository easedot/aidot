package numed

import (
	"testing"
)


func TestAdd (t *testing.T){
	var tests = []struct{
		x0,x1 *NumEd
		want *NumEd
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
		if !Equal(got,test.want){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
	}
}
func TestSub (t *testing.T){
	var tests = []struct{
		x0,x1 *NumEd
		want *NumEd
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,0,0,0)},
		{NewVec(1,2,3),NewVar(1), NewVec(0,1,2)},
		{NewVar(10),NewVec(1,2,3), NewVec(9,8,7)},
		{NewMat(2,3,1,2,3,1,2,3),NewVec(1,2,3), NewMat(2,3,0,0,0,0,0,0)},
		{NewMat(2,3,1,2,3,1,2,3),NewVar(1), NewMat(2,3,0,1,2,0,1,2)},
		{NewMat(3,2,1,2,3,1,2,3),NewMat(3,1,1,2,3), NewMat(3,2,0,1,1,-1,-1,0)},
		{NewMat(3,2,1,2,3,1,2,3),NewVar(1), NewMat(3,2,0,1,2,0,1,2)},
	}
	testFunc:= Sub
	for _,test :=range tests{
		got:=testFunc(test.x0,test.x1)
		if !Equal(got,test.want){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
	}
}
func TestMul(t *testing.T){
	var tests = []struct{
		x0,x1 *NumEd
		want *NumEd
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,1,4,9)},
		{NewVec(1,2,3),NewVar(10), NewVec(10,20,30)},
		{NewVar(10),NewVec(1,2,3), NewVec(10,20,30)},
		{NewMat(2,3,1,2,3,1,2,3),NewVec(1,2,3), NewMat(2,3,1,4,9,1,4,9)},
		{NewMat(2,3,1,2,3,1,2,3),NewVar(10), NewMat(2,3,10,20,30,10,20,30)},
		{NewMat(3,2,1,2,3,1,2,3),NewMat(3,1,1,2,3), NewMat(3,2,1,2,6,2,6,9)},
		{NewMat(3,2,1,2,3,1,2,3),NewVar(10), NewMat(3,2,10,20,30,10,20,30)},
	}
	testFunc:= Mul
	for _,test :=range tests{
		got:=testFunc(test.x0,test.x1)
		if !Equal(got,test.want){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
	}
}
func TestDiv(t *testing.T){
	var tests = []struct{
		x0,x1 *NumEd
		want *NumEd
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,1,1,1)},
		{NewVec(1,2,3),NewVar(1), NewVec(1,2,3)},
		{NewVar(10),NewVec(1,2,5), NewVec(10,5,2)},
		{NewMat(2,3,2,4,6,2,4,6),NewVec(1,2,3), NewMat(2,3,2,2,2,2,2,2)},
		{NewMat(2,3,2,4,6,2,4,6),NewVar(2), NewMat(2,3,1,2,3,1,2,3)},
		{NewMat(3,2,2,4,6,2,4,6),NewMat(3,1,1,2,2), NewMat(3,2,2,4,3,1,2,3)},
		{NewMat(3,2,2,4,6,2,4,6),NewVar(2), NewMat(3,2,1,2,3,1,2,3)},
	}
	testFunc:= Div
	for _,test :=range tests{
		got:=testFunc(test.x0,test.x1)
		if !Equal(got,test.want){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
	}
}
func TestDot(t *testing.T){
	var tests = []struct{
		x0,x1 *NumEd
		want *NumEd
	}{
		{NewMat(2,3,1,2,3,4,5,6),NewMat(2,3,1,2,3,4,5,6).T(), NewMat(2,2,14,32,32,77)},
		{NewMat(2,2,1,2,3,4),NewMat(2,2,5,6,7,8), NewMat(2,2,19,22,43,50)},
		{NewVec(1,2,3),NewVec(4,5,6).T(), NewVar(32)},
	}
	testFunc:= Dot
	for _,test :=range tests{
		got:=testFunc(test.x0,test.x1)
		if !Equal(got,test.want){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
	}
}
func TestMask(t *testing.T){
	var tests = []struct{
		x0 *NumEd
		cond EachFunc
		want *NumEd
	}{
		{NewMat(2,3,1,2,3,4,5,6),func(i,j int ,v float64) float64{
			t:=NewMat(2,3,1,2,2,4,4,6)
			if t.Get(i,j)==v{
				return 1
			}else{
				return 0
			}
		}  , NewMat(2,3,1,1,0,1,0,1)},
	}
	for _,test :=range tests{
		got:=test.x0.Mask(test.cond)
		if Equal(got,test.want){
			t.Errorf("\n %s %s %s",test.x0.Sprint("x0"),got.Sprint("y"),test.want.Sprint("w"))
		}
	}
}

func TestMax(t *testing.T){
	var tests = []struct{
		x *NumEd
		axis interface{}
		keepDims bool
		want *NumEd
	}{
		{NewMat(2,3,1,2,3,3,2,1),nil,true,NewVec(3)},
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(3,2,3)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,3,3)},
	}
	testFunc := _max
	for _,test :=range tests {
		got := testFunc(test.x,test.axis,test.keepDims)
		if Equal(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestArgMax(t *testing.T){
	var tests = []struct{
		x *NumEd
		axis interface{}
		keepDims bool
		want *NumEd
	}{
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(1,0,0)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,2,0)},
	}
	testFunc := _argmax
	for _,test :=range tests {
		got := testFunc(test.x,test.axis.(int),test.keepDims)
		if !Equal(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestMin(t *testing.T){
	var tests = []struct{
		x *NumEd
		axis interface{}
		keepDims bool
		want *NumEd
	}{
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(1,2,1)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,1,1)},
		{NewMat(2,3,1,2,3,3,2,1),nil,true,NewVec(1)},
	}
	testFunc := _min
	for _,test :=range tests {
		got := testFunc(test.x,test.axis,test.keepDims)
		if !Equal(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
func TestSum(t *testing.T){
	var tests = []struct{
		x *NumEd
		axis interface{}
		keepDims bool
		want *NumEd
	}{
		{NewMat(2,3,1,2,3,3,2,1),0,true,NewVec(4,4,4)},
		{NewMat(2,3,1,2,3,3,2,1),1,true,NewMat(2,1,6,6)},
		{NewMat(2,3,1,2,3,3,2,1),nil,true,NewVec(12)},
	}
	testFunc := _sum
	for _,test :=range tests {
		got := testFunc(test.x,test.axis,test.keepDims)
		if !Equal(got, test.want) {
			t.Errorf("\n%s\n%s\n%s", test.x.Sprint("x"), got.Sprint("y"), test.want.Sprint("w"))
		}
	}
}
