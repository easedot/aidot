package eval

import (
	"fmt"

	ng "test_ai/numgo"
)

//!+env

type Env map[Var]*ng.Variable

func (e Env) Sprint() string {
	s:=""
	for k,v:=range e{
		s+=fmt.Sprintf("%s=%s",k,v.Sprint(""))
	}
	return s
}
//!-env

//!+Eval1

func (v Var) Eval(env Env) *ng.Variable {
	return env[v]
}

func (l literal) Eval(_ Env) *ng.Variable {
	return ng.NewVar(float64(l))
}

//!-Eval1

//!+Eval2

func (u unary) Eval(env Env) *ng.Variable {
	switch u.op {
	case '+':
		return u.x.Eval(env)
	case '-':
		return ng.Neg(u.x.Eval(env))
	}
	panic(fmt.Sprintf("unsupported unary operator: %q", u.op))
}

func (b binary) Eval(env Env) *ng.Variable {
	switch b.op {
	case '+':
		return ng.Add(b.x.Eval(env),b.y.Eval(env))
	case '-':
		return ng.Sub(b.x.Eval(env),b.y.Eval(env))
	case '*':
		return ng.Mul(b.x.Eval(env),b.y.Eval(env))
	case '/':
		return ng.Div(b.x.Eval(env),b.y.Eval(env))
	}
	panic(fmt.Sprintf("unsupported binary operator: %q", b.op))
}

func (c call) Eval(env Env) *ng.Variable {
	switch c.fn {
	case "pow":
		return ng.Pow(c.args[0].Eval(env), int(c.args[1].Eval(env).At(0,0)))
	case "sin":
		return ng.Sin(c.args[0].Eval(env))
	case "cos":
		return ng.Cos(c.args[0].Eval(env))
	//case "sqrt":
	//	return ng.Sqrt(c.args[0].Eval(env))
	}
	panic(fmt.Sprintf("unsupported function call: %s", c.fn))
}

//!-Eval2
