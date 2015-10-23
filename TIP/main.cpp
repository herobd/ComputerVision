#include <opencv2/viz/vizcore.hpp>
#include <opencv2/viz/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
using namespace cv;

enum stateNames {UL_POINT, LR_POINT, VAN_POINT, DRAW_OBJ, DRAW_BACK};
stateNames state;
Point UL;
//viz::WSphere UL_dot;
Point LR;
Point3i VP;
Mat image;
double width_p;
double depth;
Affine3d freezePose;
Vec3f finalFP;

void renderAndDisplay(viz::Viz3d &window);
Vec3b biLInterp(const Mat& image, double y_c, double x_c);

void on_mouse(const viz::MouseEvent& me, void* ptr)
{
    if (me.button==viz::MouseEvent::MouseButton::LeftButton && me.type==viz::MouseEvent::Type::MouseButtonRelease)
    {
        viz::Viz3d* window = (viz::Viz3d*)ptr;
        Point3d p(me.pointer.x, me.pointer.y, 0);
        if (state==UL_POINT)
        {
            //Because we're using viz, we need to convert a
            //3D ray into the 2d point on the image
            width_p=p.x;
            Point3d orig;
            Vec3d dir;
            window->converTo3DRay(p,orig,dir);

            double t = (-orig.z)/(dir[2]);
            Point3d p2(dir[0]*t+orig.x,dir[1]*t+orig.y,dir[2]*t+orig.z);
            UL.x=p2.x;
            UL.y=p2.y;
            viz::WSphere UL_dot(p2,5,5,viz::Color::red());
            window->showWidget("UL",UL_dot);
            
        }
        else if (state==LR_POINT)
        {
            //Because we're using viz, we need to convert a
            //3D ray into the 2d point on the image
            width_p = p.x-width_p;
            Point3d orig;
            Vec3d dir;
            window->converTo3DRay(p,orig,dir);

            double t = (-orig.z)/(dir[2]);
            Point3d lr(dir[0]*t+orig.x,dir[1]*t+orig.y,1);
            LR.x=lr.x;
            LR.y=lr.y;
            viz::WSphere LR_dot(lr,5,5,viz::Color::red());
            window->showWidget("LR",LR_dot);

            //draw a rectangle around the back wall
            Point3d ul(UL.x,UL.y,1);
            Point3d ll(UL.x,LR.y,1);
            Point3d ur(LR.x,UL.y,1);
            viz::WLine top(ul,ur,viz::Color::red());
            viz::WLine left(ul,ll,viz::Color::red());
            viz::WLine right(lr,ur,viz::Color::red());
            viz::WLine bottom(ll,lr,viz::Color::red());
            window->showWidget("top_box",top);
            window->showWidget("left_box",left);
            window->showWidget("right_box",right);
            window->showWidget("bottom_box",bottom);
        }
        else if (state==VAN_POINT)
        {
            //Because we're using viz, we need to convert a
            //3D ray into the 2d point on the image
            Point3d orig;
            Vec3d dir;
            window->converTo3DRay(p,orig,dir);

            double t = (-orig.z)/(dir[2]);
            Point3d vp(dir[0]*t+orig.x,dir[1]*t+orig.y,1);
            VP.x=vp.x;
            VP.y=vp.y;
            viz::WSphere LR_dot(vp,5,5,viz::Color::red());
            window->showWidget("VP",LR_dot);
            
            //draw lines showing the perspective of the scene
            double len=fabs(orig.z/2.0);

            double ulSlope = (VP.y-UL.y)/(1.0*VP.x-UL.x);
            Point3d ul(VP.x-len,VP.y+(-len*ulSlope),1);
            double llSlope = (VP.y-LR.y)/(1.0*VP.x-UL.x);
            Point3d ll(VP.x-len,VP.y+(-len*llSlope),1);
            double urSlope = (VP.y-UL.y)/(1.0*VP.x-LR.x);
            Point3d ur(VP.x+len,VP.y+(len*urSlope),1);
            double lrSlope = (VP.y-LR.y)/(1.0*VP.x-LR.x);
            Point3d lr(VP.x+len,VP.y+(len*lrSlope),1);
            viz::WLine top(ul,vp,viz::Color::red());
            viz::WLine left(ur,vp,viz::Color::red());
            viz::WLine right(lr,vp,viz::Color::red());
            viz::WLine bottom(ll,vp,viz::Color::red());
            window->showWidget("top_mesh",top);
            window->showWidget("left_mesh",left);
            window->showWidget("right_mesh",right);
            window->showWidget("bottom_mesh",bottom);
        }
    }
    
    //This is a bit of a hack, preventing the user for changing the
    //perspective when selecting points
    if (state != DRAW_BACK)
    {
        viz::Viz3d* window = (viz::Viz3d*)ptr;
        window->setViewerPose(freezePose);
    }
}

void on_button(const viz::KeyboardEvent& me, void* ptr)
{
    //on ENTER
    if (me.code==13 && me.action==viz::KeyboardEvent::KEY_UP)
    {
        viz::Viz3d* window = (viz::Viz3d*)ptr;
        switch(state)
        {
        case UL_POINT:
            state=LR_POINT;
            break;

        case LR_POINT:
            state=VAN_POINT;
            window->removeWidget("UL");
            window->removeWidget("LR");
            break;

        case VAN_POINT:
            state=DRAW_BACK;
            


            {
            double xOff = UL.x - (UL.x-LR.x)/2.0;
            double yOff = LR.y + (UL.y-LR.y)/2.0;

            UL.x=image.size[1]/2+UL.x;
            UL.y=image.size[0]/2-UL.y;
            LR.x=image.size[1]/2+LR.x;
            LR.y=image.size[0]/2-LR.y;
            VP.x=image.size[1]/2+VP.x;
            VP.y=image.size[0]/2-VP.y;

            window->removeAllWidgets();

            Vec3f pos(0,0,image.cols);
            Vec3f fp(-xOff,-yOff,0);
            finalFP[0]=image.size[1]/2.0;
            finalFP[1]=image.size[0]/2.0;
            finalFP[2]=1.0;
            Vec3f up(0,-1,0);
            Affine3d a = viz::makeCameraPose(pos,fp,up);
            window->setViewerPose(a);

            viz::WImage3D imageN(image,image.size());
            Affine3d poseEh = Affine3d().rotate(Vec3f(CV_PI, 0.0, 0.0)).translate(fp);

            window->showWidget("imageN",imageN,poseEh);



            renderAndDisplay(*window);
            
            }
            break;

        default:
            {
//            viz::Camera c = window->getCamera();
//            std::cout << "pp:" << c.getPrincipalPoint() << " window:" << c.getWindowSize() << " fov:" << c.getFov() << " fl:" << c.getFocalLength() << " clip:" << c.getClip() << std::endl;

            }
            break;

        }


    }
}

int main(int argc, char *argv[])
{
    state=UL_POINT;

    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    viz::Viz3d window("TIP");
    window.registerMouseCallback(on_mouse,(void*)(&window));
    window.registerKeyboardCallback(on_button,(void*)(&window));

    //show  the refence image
    viz::WImage3D image2d(image,image.size());
    Affine3d poseEh = Affine3d().rotate(Vec3d(CV_PI, 0.0, 0.0));
    window.showWidget("image2d",image2d,poseEh);

    //set viewpoint
    Vec3f viewP(0,0,image.cols);
    Vec3f foc(0,0,0);
    Vec3d upr(0,-1,0);
    freezePose = viz::makeCameraPose(viewP,foc,upr);

    window.setViewerPose(freezePose);


    window.spin();
    return 1;


}

void renderAndDisplay(viz::Viz3d &window)
{
    window.removeAllWidgets();
    VP.z=1;


    int width = LR.x-UL.x;
    int height = LR.y-UL.y;



    double f  =(716.0/3000.0)*image.cols;
    double scale = width/width_p;

    depth = f*(scale-1);
    
    depth = image.cols/(0.0+width);
    depth*= (width_p/width);
    printf ("scale=%f  depth = %f\n",scale,depth);

    Vec3f botLine(0,1,-image.size[0]);
    Vec3f leftLine(1,0,0);
    Vec3f rightLine(1,0,-image.size[1]);
    Vec3f topLine(0,1,0);

//    printf("size (%d,%d)\n",image.size[0],image.size[1]);

    //The back image is simply a cutout of the original image
    Mat backImg(LR.y-UL.y,LR.x-UL.x,image.type());
    printf("size (%d,%d)\n",backImg.size[0],backImg.size[1]);
    for( int y = 0; y < backImg.rows; y++ )
    {
        for( int x = 0; x < backImg.cols; x++ )
        {
            backImg.at<Vec3b>(y,x) = image.at<Vec3b>(y+UL.y,x+UL.x);

        }
    }
    Size2d imageSize(backImg.size[1],backImg.size[0]);
    Vec3d position(0,0,0);
    Vec3d normal(1,0,0);
    Vec3d up(0,-1,0);
    viz::WImage3D back(backImg,imageSize,position,normal,up);

    Affine3d pose = Affine3d().rotate(Vec3d(0.0, CV_PI/2, 0.0));
    window.showWidget("back",back,pose);

    //The floor image uses the two reference points form the bottom of the back wall
    Mat floorImg(depth,backImg.cols,image.type());
    Vec3d position_floor(backImg.rows/2,-floorImg.rows/2,0);
    {
        Point3f one_c(UL.x,LR.y,1);
        Point3f two_c(LR.x,LR.y,1);

        Vec3f line1=one_c.cross(VP);

        //This scoring is a method of deciding which intersection point (of perspective lines and image boundary) is farthest out
        Point3f three_c;
        Point3f three_c1 = line1.cross(leftLine);
        double score1 = ((three_c1.y/three_c1.z > image.size[0])?(three_c1.y/three_c1.z -image.size[0]):0) +
                ((three_c1.x/three_c1.x < 0)?-(three_c1.x/three_c1.x):0);
        Point3f three_c2 = line1.cross(botLine);
        double score2 = ((three_c2.y/three_c2.z > image.size[0])?(three_c2.y/three_c2.z -image.size[0]):0) +
            ((three_c2.x/three_c2.x < 0)?-(three_c2.x/three_c2.x):0);
        if (score2>score1)
        {
            three_c=three_c2;
        }
        else
            three_c=three_c1;

        Vec3f line2=two_c.cross(VP);
        Point3f four_c;
        Point3f four_c1 = line2.cross(rightLine);//compute intersection
        score1 = ((four_c1.y/four_c1.z > image.size[0])?(four_c1.y/four_c1.z -image.size[0]):0) +
                ((four_c1.x/four_c1.x >image.size[1])?(four_c1.x/four_c1.x-image.size[1]):0);
        Point3f four_c2 = line2.cross(botLine);//compute intersection
        score2 = ((four_c2.y/four_c2.z > image.size[0])?(four_c2.y/four_c2.z -image.size[0]):0) +
                ((four_c2.x/four_c2.x >image.size[1])?(four_c2.x/four_c2.x-image.size[1]):0);
        if (score2>score1)
        {
            four_c=four_c2;
        }
        else
            four_c=four_c1;
//        printf("score1=%f, score2=%f\n",score1,score2);

        std::vector<Point2f> dstPoints(4);
        dstPoints[0].x=one_c.x/one_c.z;
        dstPoints[1].x=two_c.x/two_c.z;
        dstPoints[2].x=three_c.x/three_c.z;
        dstPoints[3].x=four_c.x/four_c.z;
        dstPoints[0].y=one_c.y/one_c.z;
        dstPoints[1].y=two_c.y/two_c.z;
        dstPoints[2].y=three_c.y/three_c.z;
        dstPoints[3].y=four_c.y/four_c.z;

        //    printf("dst points: (%f,%f) (%f,%f) (%f,%f) (%f,%f)\n",dstPoints[0].x,dstPoints[0].y,dstPoints[1].x,dstPoints[1].y,dstPoints[2].x,dstPoints[2].y,dstPoints[3].x,dstPoints[3].y);

        std::vector<Point2f> srcPoints(4);
        srcPoints[0].x=-width/2.0;
        srcPoints[1].x=width/2.0;
        srcPoints[2].x=-width/2.0;
        srcPoints[3].x=width/2.0;

        srcPoints[0].y=0;
        srcPoints[1].y=0;
        srcPoints[2].y=depth;
        srcPoints[3].y=depth;

        Mat floorHomog = findHomography(srcPoints,dstPoints);
        
        std::cout << "homog " << floorHomog.type() << std::endl << floorHomog << std::endl;
        


        printf("size floor (%d,%d)\n",floorImg.size[0],floorImg.size[1]);

        //actually create floor image
        for( double z_w = 0; z_w < floorImg.rows; z_w+=1.0 )
        {
            for( double x_w = -floorImg.cols/2.0; x_w < floorImg.cols/2.0; x_w+=1.0 )
            {

                int x = x_w + floorImg.cols/2.0;
                int z = z_w;
                //            Vec3f p_x(x_w,z_w,1.0);
                Mat p_x = (Mat_<double>(3,1) << x_w,z_w,1.0);

                //            std::cout << "p_x " << p_x.type() << std::endl << p_x << std::endl;

                Mat p_c = floorHomog*(p_x);
                double x_c = p_c.at<double>(0,0)/p_c.at<double>(2,0);
                double y_c = p_c.at<double>(1,0)/p_c.at<double>(2,0);

                if (x_c>=0 && x_c < image.size[1] && y_c>=0 && y_c < image.size[0])
                {

                    floorImg.at<Vec3b>(z,x) = biLInterp(image,y_c,x_c);
                }


            }
        }

        Size2d imageSize_floor(floorImg.size[1],floorImg.size[0]);
        //y,z,x
        Vec3d position_floor(backImg.rows/2,-floorImg.rows/2,0);
        viz::WImage3D floor(floorImg,floorImg.size(),position_floor,normal,up);

        Affine3d pose_floor = Affine3d().rotate(Vec3d(0.0,CV_PI/2, 0.0)).rotate(Vec3d(-CV_PI/2,0.0, 0.0));
        window.showWidget("floor",floor,pose_floor);

    }



    /////////////
    ////Cieling
    ///
    {

        Mat ceilImg(depth,width,image.type());

        Point3f one_c(UL.x,UL.y,1);
        Point3f two_c(LR.x,UL.y,1);

        Vec3f line1=one_c.cross(VP);

        //Same scoring ideas as above, just selecting better intersection point
        Point3f three_c;
        Point3f three_c1 = line1.cross(leftLine);//compute intersection
        double score1 = ((three_c1.y/three_c1.z <0)?-(three_c1.y/three_c1.z):0) +
                ((three_c1.x/three_c1.x < 0)?-(three_c1.x/three_c1.x):0);
        Point3f three_c2 = line1.cross(topLine);//compute intersection
        double score2 = ((three_c2.y/three_c2.z <0)?-(three_c2.y/three_c2.z):0) +
                ((three_c2.x/three_c2.x < 0)?-(three_c2.x/three_c2.x):0);
        if (score2>score1)
        {
            three_c=three_c2;
//            printf("choose top over right\n");
        }
        else
            three_c=three_c1;
        printf("score1=%f, score2=%f\n",score1,score2);

        Vec3f line2=two_c.cross(VP);
        Point3f four_c;
        Point3f four_c1 = line2.cross(rightLine);//compute intersection
        score1 = ((four_c1.y/four_c1.z  < 0)?-(four_c1.y/four_c1.z):0) +
                ((four_c1.x/four_c1.x >image.size[1])?(four_c1.x/four_c1.x-image.size[1]):0);
        Point3f four_c2 = line2.cross(topLine);//compute intersection
        score2 = ((four_c2.y/four_c2.z < 0)?-(four_c2.y/four_c2.z):0) +
                ((four_c2.x/four_c2.x >image.size[1])?(four_c2.x/four_c2.x-image.size[1]):0);
        if (score2>score1)
        {
            four_c=four_c2;
//            printf("choose bot over right\n");
        }
        else
            four_c=four_c1;
//        printf("score1=%f, score2=%f\n",score1,score2);

        std::vector<Point2f> dstPoints(4);
        dstPoints[0].x=one_c.x/one_c.z;
        dstPoints[1].x=two_c.x/two_c.z;
        dstPoints[2].x=three_c.x/three_c.z;
        dstPoints[3].x=four_c.x/four_c.z;
        dstPoints[0].y=one_c.y/one_c.z;
        dstPoints[1].y=two_c.y/two_c.z;
        dstPoints[2].y=three_c.y/three_c.z;
        dstPoints[3].y=four_c.y/four_c.z;

            printf("dst points: (%f,%f) (%f,%f) (%f,%f) (%f,%f)\n",dstPoints[0].x,dstPoints[0].y,dstPoints[1].x,dstPoints[1].y,dstPoints[2].x,dstPoints[2].y,dstPoints[3].x,dstPoints[3].y);

        std::vector<Point2f> srcPoints(4);
        srcPoints[0].x=-width/2;
        srcPoints[1].x=width/2;
        srcPoints[2].x=-width/2;
        srcPoints[3].x=width/2;

        srcPoints[0].y=0;
        srcPoints[1].y=0;
        srcPoints[2].y=depth;
        srcPoints[3].y=depth;


        Mat ceilHomog = findHomography(srcPoints,dstPoints);
        std::cout << "C homog " << ceilHomog.type() << std::endl << ceilHomog << std::endl;

        printf("size ceil (%d,%d)\n",ceilImg.size[0],ceilImg.size[1]);

        //fill ceiling image
        for( double z_w = 0; z_w < depth; z_w+=1.0 )
        {
            for( double x_w = -width/2.0; x_w < width/2.0; x_w+=1.0 )
            {
                int x = x_w + width/2.0;
                int z = z_w;

                Mat p_x = (Mat_<double>(3,1) << x_w,z_w,1.0);

                Mat p_c = ceilHomog*(p_x);
                double x_c = p_c.at<double>(0,0)/p_c.at<double>(2,0);
                double y_c = p_c.at<double>(1,0)/p_c.at<double>(2,0);
//                if (x_w==-floorImg.cols/2.0 && z_w<50) std::cout <<  "("<<x_c<<","<<y_c<<")"<<std::endl;
                if (x_c>=0 && x_c < image.size[1] && y_c>=0 && y_c < image.size[0])
                {

                    ceilImg.at<Vec3b>(z,x) = biLInterp(image,y_c,x_c);
                }

            }
        }

//        viz::WImage3D ceil3(floorImg,floorImg.size(),position_floor,normal,up);

        Size2d imageSize_ceil(ceilImg.size[1],ceilImg.size[0]);
        //y,z,x // ,y,
        Vec3d position_ceil(-height/2,-depth/2,0);


        viz::WImage3D ceil(ceilImg,ceilImg.size(),position_ceil,normal,up);

        Affine3d pose_ceil = Affine3d().rotate(Vec3d(0.0,CV_PI/2, 0.0)).rotate(Vec3d(-CV_PI/2,0.0, 0.0));
        window.showWidget("ceil",ceil,pose_ceil);

        Mat p_FP = (Mat_<double>(3,1) << finalFP[0],finalFP[1],1.0);

        Mat p_wFP = ceilHomog.inv()*(p_FP);
        double x_wFP = p_wFP.at<double>(0,0)/p_wFP.at<double>(2,0);
        double z_wFP = p_wFP.at<double>(1,0)/p_wFP.at<double>(2,0);

        finalFP[0]=x_wFP;
        finalFP[1]=height/2;
        finalFP[2]=z_wFP;
    }

    ////

    /////////////
    ////Left Wall
    ///
    {
        Point3f one_c(UL.x,UL.y,1);
        Point3f two_c(UL.x,LR.y,1);

        Vec3f line1=one_c.cross(VP);

        Point3f three_c;
        Point3f three_c1 = line1.cross(leftLine);
        double score1 = ((three_c1.y/three_c1.z <0)?-(three_c1.y/three_c1.z):0) +
                ((three_c1.x/three_c1.x < 0)?-(three_c1.x/three_c1.x):0);
        Point3f three_c2 = line1.cross(topLine);
        double score2 = ((three_c2.y/three_c2.z <0)?-(three_c2.y/three_c2.z):0) +
                ((three_c2.x/three_c2.x < 0)?-(three_c2.x/three_c2.x):0);
        if (score2>score1)
        {
            three_c=three_c2;
        }
        else
            three_c=three_c1;
//        printf("score1=%f, score2=%f\n",score1,score2);

        Vec3f line2=two_c.cross(VP);
        Point3f four_c;// = line2.cross(rightLine);
        Point3f four_c1 = line2.cross(leftLine);
        score1 = ((four_c1.y/four_c1.z > image.size[0])?(four_c1.y/four_c1.z -image.size[0]):0) +
                ((four_c1.x/four_c1.x < 0)?-(four_c1.x/four_c1.x):0);
        Point3f four_c2 = line2.cross(botLine);
        score2 = ((four_c2.y/four_c2.z > image.size[0])?(four_c2.y/four_c2.z -image.size[0]):0) +
                ((four_c2.x/four_c2.x < 0)?-(four_c2.x/four_c2.x):0);
        if (score2>score1)
        {
            four_c=four_c2;
//            printf("choose bot over right\n");
        }
        else
            four_c=four_c1;
//        printf("score1=%f, score2=%f\n",score1,score2);

        std::vector<Point2f> dstPoints(4);
        dstPoints[0].x=one_c.x/one_c.z;
        dstPoints[1].x=two_c.x/two_c.z;
        dstPoints[2].x=three_c.x/three_c.z;
        dstPoints[3].x=four_c.x/four_c.z;
        dstPoints[0].y=one_c.y/one_c.z;
        dstPoints[1].y=two_c.y/two_c.z;
        dstPoints[2].y=three_c.y/three_c.z;
        dstPoints[3].y=four_c.y/four_c.z;

        //    printf("dst points: (%f,%f) (%f,%f) (%f,%f) (%f,%f)\n",dstPoints[0].x,dstPoints[0].y,dstPoints[1].x,dstPoints[1].y,dstPoints[2].x,dstPoints[2].y,dstPoints[3].x,dstPoints[3].y);

        std::vector<Point2f> srcPoints(4);
        srcPoints[0].x=0;
        srcPoints[1].x=0;
        srcPoints[2].x=depth;
        srcPoints[3].x=depth;

        srcPoints[0].y=height/2;
        srcPoints[1].y=-height/2;
        srcPoints[2].y=height/2;
        srcPoints[3].y=-height/2;

        Mat leftHomog = findHomography(srcPoints,dstPoints);
        std::cout << "L homog " << leftHomog.type() << std::endl << leftHomog << std::endl;

        Mat leftImg(height,depth,image.type());
        printf("size left (%d,%d)\n",leftImg.size[0],leftImg.size[1]);

        for( double z_w = 0; z_w < leftImg.cols; z_w+=1.0 )
        {
            for( double y_w = -leftImg.rows/2.0; y_w < leftImg.rows/2.0; y_w+=1.0 )
            {
                int y = y_w + leftImg.rows/2.0;
                int z = z_w;

                Mat p_x = (Mat_<double>(3,1) << z_w,y_w,1.0);

                Mat p_c = leftHomog*(p_x);
                double x_c = p_c.at<double>(0,0)/p_c.at<double>(2,0);
                double y_c = p_c.at<double>(1,0)/p_c.at<double>(2,0);
//                if (x_w==-floorImg.cols/2.0 && z_w<50) std::cout <<  "("<<x_c<<","<<y_c<<")"<<std::endl;
                if (x_c>=0 && x_c < image.size[1] && y_c>=0 && y_c < image.size[0])
                {

                    leftImg.at<Vec3b>(y,z) = biLInterp(image,y_c,x_c);
                }

            }
        }

        Size2d imageSize_left(leftImg.size[1],leftImg.size[0]);
        //y,z,x // ,y,
        Vec3d position_left(width/2,0,depth/2);
        viz::WImage3D left(leftImg,imageSize_left,position_left,normal,up);

        Affine3d pose_left = Affine3d().rotate(Vec3d(0,0,CV_PI));
        window.showWidget("left",left,pose_left);
    }

    ////
    /////////////
    ////Right Wall
    ///
    {
        Point3f one_c(LR.x,UL.y,1);
        Point3f two_c(LR.x,LR.y,1);

        Vec3f line1=one_c.cross(VP);

        Point3f three_c;
        Point3f three_c1 = line1.cross(rightLine);
        double score1 = ((three_c1.y/three_c1.z <0)?-(three_c1.y/three_c1.z):0) +
                ((three_c1.x/three_c1.x >image.size[1])?(three_c1.x/three_c1.x-image.size[1]):0);
        Point3f three_c2 = line1.cross(topLine);
        double score2 = ((three_c2.y/three_c2.z <0)?-(three_c2.y/three_c2.z):0) +
                ((three_c2.x/three_c2.x >image.size[1])?(three_c2.x/three_c2.x-image.size[1]):0);
        if (score2>score1)
        {
            three_c=three_c2;
//            printf("choose top over right\n");
        }
        else
            three_c=three_c1;
//        printf("score1=%f, score2=%f\n",score1,score2);

        Vec3f line2=two_c.cross(VP);
        Point3f four_c;// = line2.cross(rightLine);
        Point3f four_c1 = line2.cross(rightLine);
        score1 = ((four_c1.y/four_c1.z > image.size[0])?(four_c1.y/four_c1.z -image.size[0]):0) +
                ((four_c1.x/four_c1.x >image.size[1])?(four_c1.x/four_c1.x-image.size[1]):0);
        Point3f four_c2 = line2.cross(botLine);
        score2 = ((four_c2.y/four_c2.z > image.size[0])?(four_c2.y/four_c2.z -image.size[0]):0) +
                ((four_c2.x/four_c2.x >image.size[1])?(four_c2.x/four_c2.x-image.size[1]):0);
        if (score2>score1)
        {
            four_c=four_c2;
//            printf("choose bot over right\n");
        }
        else
            four_c=four_c1;
//        printf("score1=%f, score2=%f\n",score1,score2);

        std::vector<Point2f> dstPoints(4);
        dstPoints[0].x=one_c.x/one_c.z;
        dstPoints[1].x=two_c.x/two_c.z;
        dstPoints[2].x=three_c.x/three_c.z;
        dstPoints[3].x=four_c.x/four_c.z;
        dstPoints[0].y=one_c.y/one_c.z;
        dstPoints[1].y=two_c.y/two_c.z;
        dstPoints[2].y=three_c.y/three_c.z;
        dstPoints[3].y=four_c.y/four_c.z;

            printf("dst points: (%f,%f) (%f,%f) (%f,%f) (%f,%f)\n",dstPoints[0].x,dstPoints[0].y,dstPoints[1].x,dstPoints[1].y,dstPoints[2].x,dstPoints[2].y,dstPoints[3].x,dstPoints[3].y);

        std::vector<Point2f> srcPoints(4);
        srcPoints[0].x=0;
        srcPoints[1].x=0;
        srcPoints[2].x=depth;
        srcPoints[3].x=depth;

        srcPoints[0].y=height/2;
        srcPoints[1].y=-height/2;
        srcPoints[2].y=height/2;
        srcPoints[3].y=-height/2;

        Mat rightHomog = findHomography(srcPoints,dstPoints);
        std::cout << "R homog " << rightHomog.type() << std::endl << rightHomog << std::endl;

        Mat rightImg(height,depth,image.type());
        printf("size left (%d,%d)\n",rightImg.size[0],rightImg.size[1]);

        for( double z_w = 0; z_w < rightImg.cols; z_w+=1.0 )
        {
            for( double y_w = -rightImg.rows/2.0; y_w < rightImg.rows/2.0; y_w+=1.0 )
            {
                int y = y_w + rightImg.rows/2.0;
                int z = z_w;

                Mat p_x = (Mat_<double>(3,1) << z_w,y_w,1.0);

                Mat p_c = rightHomog*(p_x);
                double x_c = p_c.at<double>(0,0)/p_c.at<double>(2,0);
                double y_c = p_c.at<double>(1,0)/p_c.at<double>(2,0);
//                if (x_w==-floorImg.cols/2.0 && z_w<50) std::cout << "("<<x_c<<","<<y_c<<")"<<std::endl;
                if (x_c>=0 && x_c < image.size[1] && y_c>=0 && y_c < image.size[0])
                {

                    rightImg.at<Vec3b>(y,z) = biLInterp(image,y_c,x_c);
                }

            }
        }

        Size2d imageSize_right(rightImg.size[1],rightImg.size[0]);
        //y,z,x // ,y,
        Vec3d position_right(-width/2,0,depth/2);
        viz::WImage3D right(rightImg,imageSize_right,position_right,normal,up);

        Affine3d pose_right = Affine3d().rotate(Vec3d(0,0,CV_PI));
        window.showWidget("right",right,pose_right);
    }
    printf("loading...\n");
    ////

    Point center;
    center.x=UL.x + (LR.x-UL.x)/2.0;
    center.y=UL.y + (LR.y-UL.y)/2.0;

    //Vec3f pos(-VP.x+center.x, -VP.y+center.y,depth);
//    Vec3f up(0,-1,0);
    //Affine3d a = viz::makeCameraPose(pos,finalFP,up);
    //window.setViewerPose(a);


//    viz::WCoordinateSystem axis(20.0);
//    window.showWidget("axis",axis);


    //window.removeWidget("image2d");
    printf("done, starting dragging!\n");
    

}

Vec3b biLInterp(const Mat& image, double y_c, double x_c)
{

//    biLInterp(image,y_c,x_c);

    Mat out;
    Size2d size(1,1);
    Point2f p(x_c,y_c);
    getRectSubPix(image,size,p,out);
    return out.at<Vec3b>(0,0);
}
