package numgo

import (
	"fmt"
	"image/color"

	"golang.org/x/exp/rand"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	nt "test_ai/tensor"
)

type UnitGrid struct {
	XX, YY, ZZ *nt.Tensor
}

func (g UnitGrid) Dims() (c, r int)   { r, c = g.XX.Shape()[0], g.XX.Shape()[1]; return c, r }
func (g UnitGrid) Z(c, r int) float64 { return g.ZZ.Get(r, c) }
func (g UnitGrid) X(c int) float64 {
	n := g.XX.Shape()
	if c < 0 || c >= n[1] {
		panic("index out of range")
	}
	return g.XX.Get(0, c)
}
func (g UnitGrid) Y(r int) float64 {
	m := g.XX.Shape()
	if r < 0 || r >= m[0] {
		panic("index out of range")
	}
	return g.YY.Get(r, 0)
}

type EdThumbnailer struct {
	color.Color
}

// Thumbnail fulfills the plot.Thumbnailer interface.
func (et EdThumbnailer) Thumbnail(c *draw.Canvas) {
	pts := []vg.Point{
		{X: c.Min.X, Y: c.Min.Y},
		{X: c.Min.X, Y: c.Max.Y},
		{X: c.Max.X, Y: c.Max.Y},
		{X: c.Max.X, Y: c.Min.Y},
	}
	poly := c.ClipPolygonY(pts)
	c.FillPolygon(et.Color, poly)

	pts = append(pts, vg.Point{X: c.Min.X, Y: c.Min.Y})
	outline := c.ClipLinesY(pts)
	c.StrokeLines(draw.LineStyle{
		Color: color.Black,
		Width: vg.Points(1),
	}, outline...)
}

func NewPlot(name, xlabel, ylabel string, wmm, hmm int) *ploted {
	p := plot.New()
	p.Title.Text = name
	p.X.Label.Text = xlabel
	p.Y.Label.Text = ylabel
	p.Add(plotter.NewGrid())
	plt := &ploted{
		p:   p,
		wmm: wmm,
		hmm: hmm,
	}
	return plt
}

type ploted struct {
	p        *plot.Plot
	wmm, hmm int
}

func (p *ploted) Plot(name string, x, y []float64, c color.Color) {
	pts := make(plotter.XYs, len(x))
	for i := 0; i < len(x); i++ {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	s, err := plotter.NewScatter(pts)
	if err != nil {
		fmt.Printf("error:%s", err)
	}
	if c == nil {
		c = color.RGBA{R: uint8(rand.Intn(255)), B: uint8(rand.Intn(255)), A: uint8(rand.Intn(255))}
	}

	p.p.Legend.Add(name, EdThumbnailer{c})
	s.GlyphStyle.Color = c
	p.p.Add(s)
}
func (p *ploted) Save(toFile string) {
	if err := p.p.Save(vg.Length(p.wmm)*vg.Millimeter, vg.Length(p.hmm)*vg.Millimeter, toFile); err != nil {
		panic(err)
	}
}
