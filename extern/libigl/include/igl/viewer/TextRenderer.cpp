// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Wenzel Jacob <wenzel@inf.ethz.ch>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifdef IGL_VIEWER_WITH_NANOGUI
#include "TextRenderer.h"
#include "TextRenderer_fonts.h"
#include <igl/project.h>

#include <nanogui/opengl.h>
#include <nanovg.h>

#include <Eigen/Dense>
#include <iostream>
#define NANOVG_GL3
#include <nanovg_gl.h>


IGL_INLINE igl::viewer::TextRenderer::TextRenderer(): ctx(nullptr) {}

IGL_INLINE int igl::viewer::TextRenderer::Init()
{
  using namespace std;
  #ifdef NDEBUG
    ctx = nvgCreateGL3(NVG_STENCIL_STROKES | NVG_ANTIALIAS);
  #else
    ctx = nvgCreateGL3(NVG_STENCIL_STROKES | NVG_ANTIALIAS | NVG_DEBUG);
  #endif

  nvgCreateFontMem(ctx, "sans", igl_roboto_regular_ttf,
                             igl_roboto_regular_ttf_size, 0);

  return 0;
}

IGL_INLINE int igl::viewer::TextRenderer::Shut()
{
  using namespace std;
  if(ctx)
    nvgDeleteGL3(ctx);
  return 0;
}

IGL_INLINE void igl::viewer::TextRenderer::BeginDraw(
  const Eigen::Matrix4f &view,
  const Eigen::Matrix4f &proj,
  const Eigen::Vector4f &_viewport,
  float _object_scale)
{
  using namespace std;
  viewport = _viewport;
  proj_matrix = proj;
  view_matrix = view;
  object_scale = _object_scale;

  Eigen::Vector2i mFBSize;
  Eigen::Vector2i mSize;

  GLFWwindow* mGLFWWindow = glfwGetCurrentContext();
  glfwGetFramebufferSize(mGLFWWindow,&mFBSize[0],&mFBSize[1]);
  glfwGetWindowSize(mGLFWWindow,&mSize[0],&mSize[1]);
  //glViewport(0,0,mFBSize[0],mFBSize[1]);

  //glClear(GL_STENCIL_BUFFER_BIT);

  /* Calculate pixel ratio for hi-dpi devices. */
  mPixelRatio = (float)mFBSize[0] / (float)mSize[0];
  nvgBeginFrame(ctx,mSize[0],mSize[1],mPixelRatio);
}

IGL_INLINE void igl::viewer::TextRenderer::EndDraw()
{
  using namespace std;
  nvgEndFrame(ctx);
}

IGL_INLINE void igl::viewer::TextRenderer::DrawText(
  Eigen::Vector3d pos, Eigen::Vector3d normal, const std::string &text)
{
  using namespace std;
  pos += normal * 0.005f * object_scale;
  Eigen::Vector3f coord = igl::project(Eigen::Vector3f(pos(0), pos(1), pos(2)),
      view_matrix, proj_matrix, viewport);

  nvgFontSize(ctx, 16*mPixelRatio);
  nvgFontFace(ctx, "sans");
  nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
  nvgFillColor(ctx, nvgRGBA(10,10,250,255));
  nvgText(ctx, coord[0]/mPixelRatio, (viewport[3] - coord[1])/mPixelRatio, text.c_str(), NULL);
}
IGL_INLINE void igl::viewer::TextRenderer::DrawText(const float &x, const float &y, const std::string &text)
{
	nvgFontSize(ctx, FontSize * mPixelRatio);
	nvgFontFace(ctx, "sans");
	nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
	//nvgFillColor(ctx, nvgRGBA(10, 10, 250, 255));
	nvgFillColor(ctx, nvgRGBA(10, 10, 10, 255));
	nvgText(ctx, x, y, text.c_str(), NULL);
}
IGL_INLINE void igl::viewer::TextRenderer::DrawHistogram(const float &x, const float &y, const float &size,
	const std::string &head, const std::string &footer, 
	Eigen::VectorXd &value){

	nanogui::Color mBackgroundColor, mForegroundColor, mTextColor;
	mBackgroundColor = nanogui::Color(20, 128);
	mForegroundColor = nanogui::Color(255, 192, 0, 128);
	mTextColor = nanogui::Color(0, 192);

	if (value.size() < 2)
		return;

	float binN = value.size();
	int size_x = size, size_y = size;
	for (size_t i = 0; i < (size_t)value.size(); i++) {
		float v = value[i];

		float x0, y0, x_step, y_step;
		x0 = x + i * size_x / binN;
		y0 = y + (1 - v) * size_y;
		x_step = size_x / binN;
		y_step = v * size_y;

		nvgBeginPath(ctx);
		nvgRect(ctx, x0, y0, x_step, y_step);
		nvgStrokeColor(ctx, nanogui::Color(100, 255));
		//nvgStrokeColor(ctx, nanogui::Color(100, 0));
		nvgStroke(ctx);
		nvgFillColor(ctx, mForegroundColor);
		nvgFill(ctx);
	}

	//float cur_x = x, cur_y = y + size_y;
	//double binsize = size_x / (value.size());
	////left part
	//nvgBeginPath(ctx);
	//nvgMoveTo(ctx, cur_x, cur_y);
	//cur_x = x;
	//cur_y = y;
	//nvgLineTo(ctx, cur_x, cur_y);

	//for (size_t i = 0; i < value.size(); i++) {
	//	cur_x = x + (i + 1) * binsize;
	//	cur_y = y + (1 - value[i + 1]) * size_y;
	//	nvgLineTo(ctx, cur_x, cur_y);
	//}
	//cur_x = x + (value.size()-1) * binsize;
	//cur_y = y + (1 - value[(value.size() - 1)]) * size_y;
	//nvgLineTo(ctx, cur_x, cur_y);
	//cur_x = x + (value.size() - 1) * binsize;
	//cur_y = y + size_y;
	//nvgLineTo(ctx, cur_x, cur_y);
	//nvgStrokeColor(ctx, nanogui::Color(100, 255));
	//nvgStroke(ctx);
	//nvgFillColor(ctx, mForegroundColor);
	//nvgFill(ctx);


	nvgFontFace(ctx, "sans");

	float fontsize = FontSize * mPixelRatio;

	if (!head.empty()) {
		nvgFontSize(ctx, fontsize);
		nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_BOTTOM);
		nvgFillColor(ctx, mTextColor);
		nvgText(ctx, x - 3, y, head.c_str(), NULL);
	}

	if (!footer.empty()) {
		nvgFontSize(ctx, fontsize);
		nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_BOTTOM);
		nvgFillColor(ctx, mTextColor);
		nvgText(ctx, x + size_x + 10, y + size_y+3, footer.c_str(), NULL);
	}

	nvgBeginPath(ctx);
	nvgRect(ctx, x, y, size_x, size_y);
	nvgStrokeColor(ctx, nanogui::Color(100, 255));
	nvgStroke(ctx);
}
IGL_INLINE void igl::viewer::TextRenderer::DrawProgress(const float &x, const float &y, const float &xsize, const float &ysize,
	const std::string &head, const std::string &footer,
	int &curpos, std::vector<double> &components, std::vector<double> &timings) {

	nanogui::Color mBackgroundColor, mForegroundColor, mTextColor, mForegroundColor2;
	mBackgroundColor = nanogui::Color(20, 128);
	mForegroundColor = nanogui::Color(224, 102, 102, 128);
	mForegroundColor2 = nanogui::Color(0, 192, 255, 128);
	mTextColor = nanogui::Color(0, 192);

	if (components.size() < 2)
		return;

	float cur_x = x, cur_y = y + ysize;
	double binsize = xsize / (components.size() - 1);
	//left part
	nvgBeginPath(ctx);
	nvgMoveTo(ctx, cur_x, cur_y);
	cur_x = x;
	cur_y = y;
	nvgLineTo(ctx, cur_x, cur_y);

	for (size_t i = 0; i < (size_t)curpos; i++) {
		cur_x = x + (i + 1) * binsize;
		cur_y = y + (1 - components[i + 1]) * ysize;
		nvgLineTo(ctx, cur_x, cur_y);
	}
	cur_x = x + curpos * binsize;
	cur_y = y + (1 - components[curpos]) * ysize;
	nvgLineTo(ctx, cur_x, cur_y);
	cur_x = x + curpos * binsize;
	cur_y = y + ysize;
	nvgLineTo(ctx, cur_x, cur_y);
	nvgStrokeColor(ctx, nanogui::Color(100, 255));
	nvgStroke(ctx);
	nvgFillColor(ctx, mForegroundColor2);
	nvgFill(ctx);
	//right part
	nvgBeginPath(ctx);
	cur_x = x + curpos * binsize;
	cur_y = y + ysize;
	nvgMoveTo(ctx, cur_x, cur_y);
	cur_x = x + curpos * binsize;
	cur_y = y + (1 - components[curpos]) * ysize;
	nvgLineTo(ctx, cur_x, cur_y);

	for (size_t i = curpos; i < components.size() - 1; i++) {
		cur_x = x + (i + 1) * binsize;
		cur_y = y + (1 - components[i + 1]) * ysize;
		nvgLineTo(ctx, cur_x, cur_y);
	}
	cur_x = x + xsize;
	cur_y = y + (1 - components[components.size() - 1]) * ysize;
	nvgLineTo(ctx, cur_x, cur_y);
	cur_x = x + xsize;
	cur_y = y + ysize;
	nvgLineTo(ctx, cur_x, cur_y);
	nvgStrokeColor(ctx, nanogui::Color(100, 255));
	nvgStroke(ctx);
	nvgFillColor(ctx, mForegroundColor);
	nvgFill(ctx);
	//dots
	float r = 4 * mPixelRatio;
	for (size_t i = 0; i < curpos + 1; i++) {
		float x0, y0, x_step, y_step;
		x0 = x + i * binsize;
		y0 = y + (1 - components[i]) * ysize;

		nvgBeginPath(ctx);
		nvgCircle(ctx, x0, y0, r);
		nvgStrokeColor(ctx, nanogui::Color(0, 255));
		nvgStroke(ctx);
		nvgFillColor(ctx, mForegroundColor2);
		nvgFill(ctx);
	}
	for (size_t i = curpos + 1; i < components.size(); i++) {
		float x0, y0, x_step, y_step;
		x0 = x + i * binsize;
		y0 = y + (1 - components[i]) * ysize;

		nvgBeginPath(ctx);
		nvgCircle(ctx, x0, y0, r);
		nvgStrokeColor(ctx, nanogui::Color(0, 255));
		nvgStroke(ctx);
		nvgFillColor(ctx, mForegroundColor);
		nvgFill(ctx);
	}

	//axis text
	nvgFontFace(ctx, "sans");
	float fontsize = FontSize * mPixelRatio;

	if (!head.empty()) {
		nvgFontSize(ctx, fontsize);
		nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_BOTTOM);
		nvgFillColor(ctx, mTextColor);
		nvgText(ctx, x - 3, y , head.c_str(), NULL);
	}

	if (!footer.empty()) {
		nvgFontSize(ctx, fontsize);
		nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_BOTTOM);
		nvgFillColor(ctx, mTextColor);
		nvgText(ctx, x + xsize + 10, y + ysize, footer.c_str(), NULL);
	}

	nvgBeginPath(ctx);
	nvgRect(ctx, x, y, xsize, ysize);
	nvgStrokeColor(ctx, nanogui::Color(100, 255));
	nvgStroke(ctx);
}
#endif
