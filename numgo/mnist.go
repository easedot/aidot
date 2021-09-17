package numgo

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"path"

	"github.com/cheggaaa/pb/v3"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	nd "test_ai/numed"
	nt "test_ai/tensor"
	ut "test_ai/utils"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28
	Height     = 28
)

var (
	ErrFormat = errors.New("mnist: invalid format")
	ErrSize   = errors.New("mnist: size mismatch")
)

type Image [Width * Height]byte

func (img *Image) ColorModel() color.Model {
	return color.GrayModel
}
func (img *Image) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{0, 0},
		Max: image.Point{Width, Height},
	}
}
func (img *Image) At(x, y int) color.Color {
	return color.Gray{Y: img[y*Width+x]}
}
func (img *Image) Set(x, y int, v byte) {
	img[y*Width+x] = v
}
func (img *Image) Slice() [][]float64 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	iaa := make([][]float64, height)
	for y := 0; y < height; y++ {
		row := make([]float64, width)
		for x := 0; x < width; x++ {
			row[x] = float64(img[y*Width+x])
		}
		iaa[y] = row
	}
	return iaa
}

type Label int8

func Mnist(train bool) *DataSet {
	c := NewCompose(func(i, j int, v float64) float64 {
		v = v / 255.0
		return v
	})
	d := DataSet{trans: c, IDataset: &mnist{DataSet{Train: train}}}
	d.Init()
	return &d
}

type mnist struct {
	DataSet
}

func (m *mnist) prePare(d *DataSet) {
	url := "http://yann.lecun.com/exdb/mnist/"
	trainFiles := map[string]string{
		"target": "train-images-idx3-ubyte.gz",
		"label":  "train-labels-idx1-ubyte.gz",
	}
	testFiles := map[string]string{
		"target": "t10k-images-idx3-ubyte.gz",
		"label":  "t10k-labels-idx1-ubyte.gz",
	}
	dataPath := fmt.Sprintf("%s%s", url, trainFiles["target"])
	labelPath := fmt.Sprintf("%s%s", url, trainFiles["label"])
	if !m.Train {
		dataPath = fmt.Sprintf("%s%s", url, testFiles["target"])
		labelPath = fmt.Sprintf("%s%s", url, testFiles["label"])
	}

	//wg := sync.WaitGroup{}
	//wg.Add(2)
	//go func() {
	d.Data, _ = m.loadData(dataPath)
	//defer wg.Done()
	//}()
	//go func() {
	d.Label = m.loadLabel(labelPath)
	//defer wg.Done()
	//}()
	//wg.Wait()
}
func (m *mnist) loadLabel(filePath string) []float64 {
	filename, _, err := fetch(filePath)
	if err != nil {
		panic(err)
	}
	//todo 这里可以改成多线程处理，边取边转换flatten
	labels, _ := readLabelsFile(filename)
	n := len(labels)
	t := make([]float64, n)

	for i, l := range labels {
		t[i] = float64(l)
	}
	return t
}

func (m *mnist) loadData(filePath string) (*nt.Tensor, error) {
	filename, _, err := fetch(filePath)
	if err != nil {
		panic(err)
	}
	//todo 这里可以改成多线程处理，边取边转换flatten
	images, e := readImagesFile(filename)
	if e != nil {
		return nil, e
	}

	n := len(images)
	t := make([][]float64, n)

	for i, v := range images {
		//2dims to1dims 28X28 to 784
		_, _, d := ut.Flatten(v.Slice())
		t[i] = d
	}
	dr, dc, dd := ut.Flatten(t)
	d := nt.NewData(dd, dr, dc)
	return d, nil
}
func (m *mnist) show(r, c int) {

}

func fetch(url string) (filename string, n int64, err error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()

	local := fmt.Sprintf("data/%s", path.Base(resp.Request.URL.Path))
	if _, err := os.Stat(local); os.IsNotExist(err) {
		f, err := os.Create(local)
		if err != nil {
			return "", 0, err
		}
		bar := pb.Full.Start64(resp.ContentLength)
		//bar := progressbar.DefaultBytes(
		//	resp.ContentLength,
		//	"downloading",
		//)
		barReader := bar.NewProxyWriter(f)
		n, err = io.Copy(barReader, resp.Body)
		// Close file, but prefer error from Copy, if any.
		if closeErr := f.Close(); err == nil {
			err = closeErr
		}
		return local, n, err
	} else {
		return local, 0, nil
	}
}

type imageFileHeader struct {
	Magic     int32
	NumImages int32
	Height    int32
	Width     int32
}
type labelFileHeader struct {
	Magic     int32
	NumLabels int32
}

func readImage(r io.Reader) (*Image, error) {
	img := &Image{}
	err := binary.Read(r, binary.BigEndian, img)
	return img, err
}
func readImagesFile(path string) ([]*Image, error) {
	file, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer file.Close()

	//st, _ := file.Stat()
	//bar := pb.Full.Start64(-1)
	//breader := bar.NewProxyReader(file)
	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	header := imageFileHeader{}
	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}
	if header.Magic != imageMagic ||
		header.Width != Width ||
		header.Height != header.Height {
		return nil, ErrFormat
	}
	images := make([]*Image, header.NumImages)
	for i := int32(0); i < header.NumImages; i++ {
		images[i], err = readImage(reader)
		if err != nil {
			return nil, err
		}
	}
	return images, nil
}

func readLabelsFile(path string) ([]Label, error) {
	file, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer file.Close()
	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}

	header := labelFileHeader{}

	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	if header.Magic != labelMagic {
		return nil, err
	}

	labels := make([]Label, header.NumLabels)
	for i := int32(0); i < header.NumLabels; i++ {
		err = binary.Read(reader, binary.BigEndian, &labels[i])
		if err != nil {
			return nil, err
		}
	}
	return labels, nil
}

func PrintImage(image [][]uint8) {
	for _, row := range image {
		for _, pix := range row {
			if pix == 0 {
				fmt.Print(" ")
			} else {
				fmt.Printf("%X", pix/16)
			}
		}
		fmt.Println()
	}
}
func PrintImageF(image []float64, tt int) {
	fmt.Printf("img:%d", tt)
	for i, pix := range image {
		if pix == 0 {
			fmt.Print(" ")
		} else {
			fmt.Printf("%X", uint8(pix)/16)
		}
		if i%28 == 0 {
			fmt.Println()
		}
	}
}

//func PrintI(image []float64, tt int) {
//	x := nt.NewArange(0, 27, 1)
//	y := nt.NewArange(0, 27, 1)
//	xx, yy := nd.MeshGrid(x, y)
//	data := nt.NewVec(image...)
//	data = data.Reshape(28, 28)
//	zz := nt.NewZeros(28, 28)
//	for i := 0; i < 28; i++ {
//		for j := 0; j < 28; j++ {
//			fmt.Printf("i:%d,j%d", i, j)
//			zz.Set(data.Get(i, j), i, j)
//		}
//	}
//	ug := UnitGrid{xx, yy, zz}
//	var pls []plot.Plotter
//	c := plotter.NewContour(
//		ug,
//		nil,
//		palette.Rainbow(10, palette.Blue, palette.Red, 1, 1, 1),
//	)
//	pls = append(pls, c)
//	p := plot.New()
//	p.Title.Text = "Plotutil example"
//	p.X.Label.Text = "X"
//	p.Y.Label.Text = "Y"
//	p.Add(plotter.NewGrid())
//
//	p.Add(pls...)
//	if err := p.Save(10*vg.Inch, 6*vg.Inch, fmt.Sprintf("./temp/%d.png", tt)); err != nil {
//		panic(err)
//	}
//
//}

func PrintS(image []float64, tt int) {
	data := nd.NewVec(image...)
	data = data.Reshape(28, 28)

	pts := make(plotter.XYs, 0)
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			if data.Get(i, j) > 0 {
				pt := plotter.XY{X: float64(i), Y: float64(j)}
				pts = append(pts, pt)
			}
		}
	}

	var pls []plot.Plotter
	s, err := plotter.NewScatter(pts)
	if err != nil {
		panic(err)
	}
	pls = append(pls, s)
	p := plot.New()
	p.Title.Text = "Plotutil example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	p.Add(pls...)
	if err := p.Save(6*vg.Inch, 6*vg.Inch, fmt.Sprintf("./temp/%d.png", tt)); err != nil {
		panic(err)
	}

}

func PrintImg(i []float64, tt int) {
	println(tt)
	data := nd.NewVec(i...)
	data = data.Reshape(28, 28)

	// Create an 100 x 50 image
	img := image.NewRGBA(image.Rect(0, 0, 28, 28))
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			co := uint8(data.Get(i, j))
			img.Set(i, j, color.RGBA{co, co, co, 255})
		}
	}

	// Draw a red dot at (2, 3)

	// Save to out.png
	f, _ := os.OpenFile(fmt.Sprintf("./temp/%d.png", tt), os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	png.Encode(f, img)
}
