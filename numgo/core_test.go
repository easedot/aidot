package numgo

import (
	"testing"

	"gonum.org/v1/gonum/mat"
	ut "test_ai/utils"
)

func TestAdd (t *testing.T){
	var tests = []struct{
		x0,x1 *Variable
		want *Variable
		g0,g1 *Variable
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,2,4,6),NewVec(1,1,1),NewVec(1,1,1)},
		{NewVec(1,2,3),NewVar(10), NewVec(11,12,13),NewVec(1,1,1),NewVar(3)},
		{NewVar(10),NewVec(1,2,3), NewVec(11,12,13),NewVar(3),NewVec(1,1,1)},
		{NewMat(2,3,1,2,3,1,2,3),NewVec(1,2,3), NewMat(2,3,2,4,6,2,4,6),NewMat(2,3,1,1,1,1,1,1),NewVec(2,2,2)},
		{NewMat(2,3,1,2,3,1,2,3),NewVar(10), NewMat(2,3,11,12,13,11,12,13),NewMat(2,3,1,1,1,1,1,1),NewVar(6)},
		{NewMat(3,2,1,2,3,1,2,3),NewMat(3,1,1,2,3), NewMat(3,2,2,3,5,3,5,6),NewMat(3,2,1,1,1,1,1,1),NewMat(3,1,2,2,2)},
		{NewMat(3,2,1,2,3,1,2,3),NewVar(10), NewMat(3,2,11,12,13,11,12,13),NewMat(3,2,1,1,1,1,1,1),NewVar(6)},
	}
	for _,test :=range tests{
		got:=add(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		g0,g1:=test.x0.Grad,test.x1.Grad
		dcf0:= func(x... *Variable)*Variable {
			return add(x[0],test.x1)
		}
		dc0:=NumericalDiff(dcf0,test.x0)
		dcf1:= func(x... *Variable)*Variable {
			return add(x[0],test.x0)
		}
		dc1:=NumericalDiff(dcf1,test.x1)

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
		g0,g1 *Variable
	}{
		{NewVec(1,2,3),NewVec(1,2,3), NewMat(1,3,1,4,9),NewVec(1,1,1),NewVec(1,1,1)},
		{NewVec(1,2,3),NewVar(10), NewVec(10,20,30),NewVec(1,1,1),NewVar(3)},
	}
	for _,test :=range tests{
		got:=mul(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		//got.Backward(true)
		//g0,g1:=test.x0.Grad,test.x1.Grad
		////got.Plot(true,"/Users/haihui/Downloads/add_y.png")
		////g0.Plot(true,"/Users/haihui/Downloads/add_g0.png")
		////g1.ClearGrade()
		////g1.Backward(true)
		////g1.Plot(true,"/Users/haihui/Downloads/add_g1.png")
		//if !mat.Equal(g0.Data,test.g0.Data){
		//	t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g0.Sprint("g0"),test.g0.Sprint("wg0"))
		//}
		//if !mat.Equal(g1.Data,test.g1.Data){
		//	t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g1.Sprint("g1"),test.g1.Sprint("wg1"))
		//}

	}

}

func TestMatmul(t *testing.T){
	var tests = []struct{
		x0,x1 *Variable
		want *Variable
		g0,g1 *Variable
	}{
		{NewMat(2,2,1,2,3,4),NewMat(2,2,5,6,7,8), NewMat(2,2,19,22,43,50),NewVec(1,1,1),NewVec(1,1,1)},
	}
	for _,test :=range tests{
		got:=matmul(test.x0,test.x1)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),got.Sprint("y"),test.want.Sprint("w"))
		}
		//got.Backward(true)
		//g0,g1:=test.x0.Grad,test.x1.Grad
		////got.Plot(true,"/Users/haihui/Downloads/add_y.png")
		////g0.Plot(true,"/Users/haihui/Downloads/add_g0.png")
		////g1.ClearGrade()
		////g1.Backward(true)
		////g1.Plot(true,"/Users/haihui/Downloads/add_g1.png")
		//if !mat.Equal(g0.Data,test.g0.Data){
		//	t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g0.Sprint("g0"),test.g0.Sprint("wg0"))
		//}
		//if !mat.Equal(g1.Data,test.g1.Data){
		//	t.Errorf("\n %s %s %s %s",test.x0.Sprint("x0"),test.x1.Sprint("x1"),g1.Sprint("g1"),test.g1.Sprint("wg1"))
		//}

	}

}

func TestSum (t *testing.T){
	var tests = []struct{
		input *Variable
		want *Variable
	}{
		{NewVec(1,2,3,4,5,6), NewVar(21)},
	}
	for _,test :=range tests{
		got:=sum(test.input)
		if !mat.Equal(got.Data,test.want.Data){
			t.Errorf("\n %s %s %s",test.input.Sprint("x"),got.Sprint("y"),test.want.Sprint("w"))
		}
		got.Backward(true)
		test.input.Print("g")
	}
}

func TestForward(t *testing.T){
	//tt:=[][]float64{
	//	{1,2,3,4},
	//	{2,3,4,5},
	//}
	//println(len(tt[len(tt)-1]))
	m := mat.NewDense(2, 2, []float64{
		4.0,4.0,
		4.0,4.0,
	})

	r := mat.NewDense(2, 2, []float64{
		32.0,32.0,
		32.0,32.0,
	})
	x:=&Variable{Data: m}
	if got:=pow(x,2);!mat.Equal(got.Data,r){
		t.Errorf("Square error\n x:\n%s y:\n%s ",ut.SprintDense("x",x.Data),ut.SprintDense("y",got.Data))
	}
}
func TestBackward(t *testing.T){
	//m := mat.NewDense(1, 1, []float64{
	//	2.0,
	//})
	m := mat.NewDense(2, 2, []float64{
		2.0,2.0,
		2.0,2.0,
	})
	f:= func (x...*Variable) *Variable {
		y := mul(x[0],x[0])
		return y
	}
	x:=&Variable{Data: m}
	//f:=NewFunction(&Pow{C:2})
	nd :=NumericalDiff(f,x)
	ut.PrintDense("dg",nd.Data)
	y:=f(x)
	ut.PrintDense("yd",y.Data)
	y.Backward(true)
	ut.PrintDense("xg",x.Grad.Data)
	if !mat.EqualApprox(x.Grad.Data, nd.Data,1e-4){
		t.Errorf("Square error\n xg:\n%s dg:\n%s ",ut.SprintDense("x",x.Grad.Data),ut.SprintDense("y",nd.Data))
	}
}
func TestMatrixVectorMul(t *testing.T) {
	a := mat.NewDense(3, 3, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
	})
	b := mat.NewVecDense(3, []float64{
		1, 2, 3,
	})
	actual := make([]float64, 3)
	c := mat.NewVecDense(3, actual)

	// this was the method, I was looking for.
	c.MulVec(a, b)
	//ut.PrintDense("x",c)
	//expected := []float64{14, 32, 50}
	//assert.Equal(t, expected, actual)
}

func TestPolt(t *testing.T){
	//com:= func (x...*Variable) *Variable {
	//	a := mul(x[0],x[0])
	//	b := exp(a)
	//	y := mul(b,b)
	//	return y
	//}
	com:= func (x...*Variable) *Variable {
		a := mul(x[0],x[0])
		b := exp(a)
		y := add(b,b)
		return y
	}

	//m := mat.NewDense(3, 3, []float64{
	//	1,2,3,
	//	3,2,1,
	//	1,2,3,
	//})
	//x := &Variable{Data:m}
	x:=NewVar(0.5)
	y := com(x,x)

	dy := NumericalDiff(com,x)
	ut.PrintDense("dg",dy.Data)

	y.Backward(true)
	gx:=x.Grad
	ut.PrintDense("gx",gx.Data)

	x.ClearGrade()
	gx.Backward(true)
	gx2:=x.Grad
	ut.PrintDense("gx2",gx2.Data)
	PlotDotGraph(gx2,true,"/Users/haihui/Downloads/pow.png")
}
