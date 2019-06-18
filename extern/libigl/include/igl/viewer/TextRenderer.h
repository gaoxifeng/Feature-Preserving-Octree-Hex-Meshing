// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Wenzel Jacob <wenzel@inf.ethz.ch>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGL_VIEWER_TEXT_RENDERER_H
#define IGL_VIEWER_TEXT_RENDERER_H
#ifdef IGL_VIEWER_WITH_NANOGUI

#include <Eigen/Dense>

#include <igl/igl_inline.h>
#include <map>
#include <vector>

struct NVGcontext;

namespace igl
{
namespace viewer
{

  class TextRenderer
  {
  public:
    IGL_INLINE TextRenderer();

    IGL_INLINE virtual int Init();
    IGL_INLINE virtual int Shut();

    IGL_INLINE void BeginDraw(const Eigen::Matrix4f &view,const Eigen::Matrix4f &proj,
      const Eigen::Vector4f &_viewport,float _object_scale);

    IGL_INLINE void EndDraw();

    IGL_INLINE void DrawText(Eigen::Vector3d pos,Eigen::Vector3d normal,const std::string &text);
	IGL_INLINE void DrawText(const float &x, const float &y, const std::string &text);
	IGL_INLINE void DrawHistogram(const float &x, const float &y, const float &size,
		const std::string &head, const std::string &footer,
		Eigen::VectorXd &value);

	IGL_INLINE void DrawProgress(const float &x, const float &y, const float &xsize, const float &ysize,
		const std::string &head, const std::string &footer,
		int &curpos, std::vector<double> &components, std::vector<double> &timings);

	IGL_INLINE void setFontsize(const float &fs) { FontSize = fs; };

  protected:
    std::map<std::string,void *> m_textObjects;
    Eigen::Matrix4f view_matrix,proj_matrix;
    Eigen::Vector4f viewport;
    float object_scale;
    float mPixelRatio;
    NVGcontext *ctx;
	float FontSize = 32;
  };

}
}

#ifndef IGL_STATIC_LIBRARY
#  include "TextRenderer.cpp"
#endif

#endif
#endif
