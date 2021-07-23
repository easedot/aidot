package utils

import (
	"reflect"
	"testing"
)

func TestSelRowInt(t *testing.T) {
	var tests = []struct{
		x []int
		s []int
		want []int
	}{
		{[]int{2,4,-1,0,1,2,2,0,1,-1},[]int{3,0},[]int{0,2}},
		{[]int{2,4,-1,0,1,2,2,0,1,-1},[]int{4,6},[]int{1,2}},
	}
	testFunc:= SelRowInt
	for _,test :=range tests {
		got := testFunc(test.x, test.s...)
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("\n%v\n%v\n%v", test.x, got, test.want)
		}
	}
}
