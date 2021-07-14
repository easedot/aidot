package numgo

import (
	"gonum.org/v1/gonum/mat"
	ut "test_ai/utils"
)

func Sin(x *Variable)*Variable{
	f:=NewFunction(&sinFunc{})
	y:=f.Run(x)
	return y
}

type sinFunc struct {
	Function
}
func (s *sinFunc) forward(i []*Variable) []*Variable  {
	x:=i[0]
	y:=ut.LikeZeros(x.Data)
	y.Apply(ut.SinFunc,x.Data)
	return []*Variable{{Data:y}}
}
func (s *sinFunc) backward(i,o,gy []*Variable) []*Variable  {
	x:=i[0]
	g:=gy[0]
	y:=Mul(Cos(x),g)
	return []*Variable{y}
}

func Cos(x *Variable)*Variable{
	f:=NewFunction(&cosFunc{})
	y:=f.Run(x)
	return y
}
type cosFunc struct {
	Function
}
func (s *cosFunc) forward(i []*Variable) []*Variable  {
	x:=i[0]
	y:=ut.LikeZeros(x.Data)
	y.Apply(ut.CosFunc,x.Data)
	return []*Variable{{Data:y}}
}
func (s *cosFunc) backward(i,o,gy []*Variable) []*Variable  {
	x:=i[0]
	g:=gy[0]
	y:=Mul(g,Neg(Sin(x)))
	return []*Variable{y}
}

func Tanh(x *Variable)*Variable{
	f:=NewFunction(&tanhFunc{})
	y:=f.Run(x)
	return y
}
type tanhFunc struct {
	Function
}
func (s *tanhFunc) forward(i []*Variable) []*Variable  {
	x:=i[0]
	y:=ut.LikeZeros(x.Data)
	y.Apply(ut.TanghFunc,x.Data)
	return []*Variable{{Data:y}}
}
func (s *tanhFunc) backward(i,o,gy []*Variable) []*Variable  {
	y:=o[0]
	g:=gy[0]
	gx:=Mul(g,Neg(Sub(NewVar(1),Mul(y,y))))
	return []*Variable{gx}
}



func Reshape(x *Variable,s *Shape)*Variable{
	f:=NewFunction(&reshapeFunc{S: s})
	y:=f.Run(x)
	return y
}
type reshapeFunc struct {
	Function
	S *Shape
	Xs *Shape
}
func (s *reshapeFunc) forward(ix[]*Variable)[]*Variable {
	x:=ix[0]
	s.Xs=x.Shape()
	return []*Variable{{Data:mat.NewDense(s.S.R, s.S.C, x.Data.RawMatrix().Data)}}
}
func (s *reshapeFunc) backward(i,o,gy []*Variable)[]*Variable {
	return []*Variable{Reshape(gy[0],s.Xs)}
}

func Transpose(x *Variable)*Variable{
	f:=NewFunction(&transposeFunc{})
	y:=f.Run(x)
	return y
}
type transposeFunc struct {
	Function
}
func (t *transposeFunc) forward(ix[]*Variable)[]*Variable{
	x:=ix[0]
	return []*Variable{_tranposeTo(x)}
}
func (t *transposeFunc) backward(i,o,gy []*Variable)[]*Variable{
	y:=gy[0]
	return []*Variable{Transpose(y)}
}

func Exp(x ...*Variable)*Variable{
	f:=NewFunction(&expFunc{})
	return f.Run(x...)
}
type expFunc struct {
	Function
}
func (e *expFunc)forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Exp(i[0].Data)
	return [] *Variable{{Data:&o}}
}
func (e *expFunc)backward(i,o, gy []*Variable) []*Variable {
	y:=o[0]
	g:=gy[0]
	gx:= Mul(g,y)
	return [] *Variable{gx}
}

func Sum(x *Variable)*Variable{
	f:=NewFunction(&sumFunc{})
	y:=f.Run(x)
	return y
}
type sumFunc struct {
	Function
	Xs *Shape
}
func (s *sumFunc) forward(ix[]*Variable)[]*Variable{
	x:=ix[0]
	s.Xs=x.Shape()
	y:=mat.Sum(x.Data)
	r:=NewVar(y)
	return []*Variable{r}
}
func (s *sumFunc) backward(i,o,gy []*Variable)[]*Variable{
	gx:=BroadCastTo(gy[0],s.Xs)
	return []*Variable{gx}
}

func SumTo(x *Variable,s *Shape)*Variable{
	f:=NewFunction(&sumToFunc{S: s})
	y:=f.Run(x)
	return y
}
type sumToFunc struct {
	Function
	S *Shape
	Xs *Shape
}
func (s *sumToFunc) forward(ix[]*Variable)[]*Variable{
	x:=ix[0]
	s.Xs=x.Shape()
	y:= _sumTo(x,s.S)
	return []*Variable{y}
}
func (s *sumToFunc) backward(i,o,gy []*Variable)[]*Variable{
	g:=gy[0]
	gx:=BroadCastTo(g,s.Xs)
	return []*Variable{gx}
}

func BroadCastTo(x *Variable,s *Shape)*Variable{
	f:=NewFunction(&broadcastToFunc{S: s})
	y:=f.Run(x)
	return y
}
type broadcastToFunc struct {
	Function
	S *Shape
	Xs *Shape
}
func (s *broadcastToFunc) forward(ix[]*Variable)[]*Variable{
	s.Xs =ix[0].Shape()
	y:=ix[0]
	if s.S.B(s.Xs){
		y=_broadcastTo(ix[0],s.S)
	}
	return []*Variable{y}
}
func (s *broadcastToFunc) backward(i,o,gy []*Variable)[]*Variable{
	gx:=gy[0]
	if s.S.B(s.Xs){
		gx= SumTo(gy[0],s.Xs)
	}
	return []*Variable{gx}
}

func Matmul(x,w *Variable)*Variable{
	f:=NewFunction(&matMulFunc{})
	y:=f.Run(x,w)
	return y
}
type matMulFunc struct {
	Function
}
func (s *matMulFunc) forward(ix[]*Variable)[]*Variable{
	x,W:=ix[0],ix[1]
	y:=mat.Dense{}
	y.Mul(x.Data,W.Data)
	return []*Variable{{Data: &y}}
}
func (s *matMulFunc) backward(i,o,gy []*Variable)[]*Variable{
	x,W:=i[0],i[1]
	g:=gy[0]
	gx:=Matmul(g,W.T())
	gW:=Matmul(x.T(),g)
	return []*Variable{gx,gW}
}


type linearFunc struct {
	Function
}

func (l *linearFunc) forward(ix[]*Variable)[]*Variable{
	x,W:=ix[0],ix[1]
	y:= Matmul(x,W)
	var b *Variable
	if len(ix)>2{
		b=ix[2]
	}
	if b!=nil{
		y= Add(y,b)
	}
	return []*Variable{y}
}

func (l *linearFunc) backward(is, os, gys []*Variable)[]*Variable{
	gy := gys[0]
	x,W:= is[0], is[1]
	var b *Variable
	if len(is)>2{
		b=is[2]
	}
	gb:=&Variable{}
	if b!=nil{
		gb= SumTo(gy,b.Shape())
	}
	gx:=Matmul(gy,W.T())
	gW:=Matmul(x.T(), gy)
	return []*Variable{gx,gW,gb}
}
func Linear(x,W,b *Variable)*Variable{
	f:=NewFunction(&linearFunc{})
	if b==nil{
		return f.Run(x, W)
	}else{
		return f.Run(x, W, b)
	}
}

type sigmoid struct {
	Function
}

func (s *sigmoid) forward(ix []*Variable) []*Variable {
	x:=ix[0]
	y:=Add(Mul(Tanh(Mul(x,NewVar(0.5))),NewVar(0.5)),NewVar(0.5))
	return []*Variable{y}
}
func (s *sigmoid) backward(is,os,gys []*Variable) []*Variable {
	y:=os[0]
	gy:=gys[0]
	gx :=Mul(Mul(gy,y),Sub(NewVar(1),y))
	return []*Variable{gx}
}
func Sigmoid(x *Variable)*Variable  {
	f := NewFunction(&sigmoid{})
	return f.Run(x)
}


type meanSquaredError struct {
	Function
}
func (m *meanSquaredError) forward(ix []*Variable) []*Variable {
	x0,x1:=ix[0],ix[1]
	diff:=Sub(x0,x1)
	y:=Div(Sum(Mul(diff,diff)),NewVar(float64(diff.Shape().R)))
	return []*Variable{y}
}
func (m *meanSquaredError) backward(is,os,gys []*Variable) []*Variable {
	gy:=gys[0]
	x0,x1:=is[0],is[1]
	diff:=Sub(x0,x1)
	gx0:=Mul(Mul(gy,diff),Div(NewVar(2),NewVar(float64(diff.Shape().R))))
	gx1:=Neg(gx0)
	return []*Variable{gx0,gx1}
}

func MeanSquaredError(x0,x1 *Variable)*Variable{
	f := NewFunction(&meanSquaredError{})
	return f.Run(x0,x1)
}

