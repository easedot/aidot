// Copyright © 2016 Alan A. A. Donovan & Brian W. Kernighan.
// License: https://creativecommons.org/licenses/by-nc-sa/4.0/

package eval

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
	ng "test_ai/numgo"
)

//!+Eval
func TestEval(t *testing.T) {
	tests := []struct {
		expr string
		env  Env
		want *ng.Variable
	}{
		//{"sqrt(A / pi)", Env{"A": ng.NewVar(87616), "pi": ng.NewVar(math.Pi)}, ng.NewVar(167)},
		{"pow(x, 3) + pow(y, 3)", Env{"x": ng.NewVar(12), "y": ng.NewVar(1)}, ng.NewVar(1729)},
		{"pow(x, 3) + pow(y, 3)", Env{"x": ng.NewVar(9), "y": ng.NewVar(10)}, ng.NewVar(1729)},
		{"5 / 9 * (F - 32)", Env{"F": ng.NewVar(-40)}, ng.NewVar(-40)},
		{"5 / 9 * (F - 32)", Env{"F": ng.NewVar(32)}, ng.NewVar(0)},
		{"5 / 9 * (F - 32)", Env{"F": ng.NewVar(212)}, ng.NewVar(100)},
		//!-Eval
		// additional tests that don't appear in the book
		{"-1 + -x", Env{"x": ng.NewVar(1)}, ng.NewVar(-2)},
		{"-1 - x", Env{"x": ng.NewVar(1)}, ng.NewVar(-2)},
		//!+Eval
	}
	var prevExpr string
	for _, test := range tests {
		// Print expr only when it changes.
		if test.expr != prevExpr {
			fmt.Printf("\n%s\n", test.expr)
			prevExpr = test.expr
		}
		expr, err := Parse(test.expr)
		if err != nil {
			t.Error(err) // parse error
			continue
		}
		got := expr.Eval(test.env)

		if mat.Equal(got.Data,test.want.Data) {
			t.Errorf("%s.Eval() in %s got %s want %s\n",
				test.expr, test.env.Sprint(), got.Sprint(""), test.want.Sprint(""))
		}
	}
}

//!-Eval

/*
//!+output
sqrt(A / pi)
	map[A:87616 pi:3.141592653589793] => 167

pow(x, 3) + pow(y, 3)
	map[x:12 y:1] => 1729
	map[x:9 y:10] => 1729

5 / 9 * (F - 32)
	map[F:-40] => -40
	map[F:32] => 0
	map[F:212] => 100
//!-output

// Additional outputs that don't appear in the book.

-1 - x
	map[x:1] => -2

-1 + -x
	map[x:1] => -2
*/

func TestErrors(t *testing.T) {
	for _, test := range []struct{ expr, wantErr string }{
		{"x % 2", "unexpected '%'"},
		{"math.Pi", "unexpected '.'"},
		{"!true", "unexpected '!'"},
		{`"hello"`, "unexpected '\"'"},
		{"log(10)", `unknown function "log"`},
		{"sqrt(1, 2)", "call to sqrt has 2 args, want 1"},
	} {
		expr, err := Parse(test.expr)
		if err == nil {
			vars := make(map[Var]bool)
			err = expr.Check(vars)
			if err == nil {
				t.Errorf("unexpected success: %s", test.expr)
				continue
			}
		}
		fmt.Printf("%-20s%v\n", test.expr, err) // (for book)
		if err.Error() != test.wantErr {
			t.Errorf("got error %s, want %s", err, test.wantErr)
		}
	}
}

/*
//!+errors
x % 2               unexpected '%'
math.Pi             unexpected '.'
!true               unexpected '!'
"hello"             unexpected '"'

log(10)             unknown function "log"
sqrt(1, 2)          call to sqrt has 2 args, want 1
//!-errors
*/
