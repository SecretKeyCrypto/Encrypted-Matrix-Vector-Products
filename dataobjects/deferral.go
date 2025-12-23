package dataobjects

import "context"
import "C"

type DeferralFrame struct {
	stack []func()
}

type DeferralStack struct {
	stack []DeferralFrame
}

func (f *DeferralFrame) Defer(fn func()) {
	f.stack = append(f.stack, fn)
}

func (f *DeferralFrame) internalClose() {
	for len(f.stack) > 0 {
		last := len(f.stack) - 1
		f.stack[last]()
		f.stack = f.stack[:last]
	}
}

func (f *DeferralFrame) Close() {
	if !USE_FAST_CODE_WITH_CUDA {
		f.internalClose()
	}
}

func (s *DeferralStack) WithFrame() *DeferralFrame {
	s.stack = append(s.stack, DeferralFrame{stack: []func(){}})
	last := len(s.stack) - 1
	return &s.stack[last]
}

func (s *DeferralStack) Close() {
	for len(s.stack) > 0 {
		last := len(s.stack) - 1
		s.stack[last].internalClose()
		s.stack = s.stack[:last]
	}
}

type DeferralContext struct {
	ctx   *DoContext
	stack DeferralStack
}

type DeferralContextKey struct{}

var deferralContextKey = DeferralContextKey{}

func MakeDeferralContext(parent context.Context) context.Context {
	deferralContext := &DeferralContext{
		ctx: NewDoContext(),
		stack: DeferralStack{
			stack: []DeferralFrame{},
		},
	}
	return context.WithValue(parent, deferralContextKey, deferralContext)
}

func MakeDeferralContextDefault() context.Context {
	return MakeDeferralContext(context.Background())
}

func CloseDeferralContext(ctx context.Context) {
	GetDeferralStack(ctx).Close()
	FreeDoContext(GetDeferralDoContext(ctx))
}

func GetDeferralDoContext(ctx context.Context) *DoContext {
	val := ctx.Value(deferralContextKey)
	if ctx, ok := val.(*DeferralContext); ok {
		return ctx.ctx
	}
	return nil
}

func GetDeferralStack(ctx context.Context) *DeferralStack {
	val := ctx.Value(deferralContextKey)
	if ctx, ok := val.(*DeferralContext); ok {
		return &ctx.stack
	}
	return nil
}

func MakeDeferralFrame(ctx context.Context) *DeferralFrame {
	stack := GetDeferralStack(ctx)
	if stack == nil {
		return nil
	}
	return stack.WithFrame()
}
