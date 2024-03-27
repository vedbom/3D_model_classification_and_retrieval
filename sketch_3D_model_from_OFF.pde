import peasy.*;

// OFF: Object File Format
// OFF files are used to represent the geometry of a model by specifying the polygons of the model's surface
// they start with "OFF" in line 1 and then the number of vertices, number of faces and number of edges (ignore this) on line 2
// from line 3 the x, y, z coordinates of all the vertices are written
// after the vertices the face information is written in the format (number of vertices, index of vertex 1, index of vertex 2, etc.)
// the indices of the vertices start at 0 and indicate the position of the vertex in the vertices list

Model m;
PeasyCam cam;
PGraphics pg;

// the number of segments that theta=2*PI and phi=PI will split into
int numTheta = 8;
int numPhi = 9;

void setup(){
  size(500, 400, P3D);
  String[] classes = {"bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"};
  int[] trainSamples = {156, 615, 989, 286, 286, 565, 286, 780, 492, 444};
  for(int i = 0; i < classes.length; i++){
    for(int j = 1; j <= trainSamples[i]; j++){
      // measure the amount of time it takes to process one file
      int currentTime = millis();
      
      String filePath = String.format("ModelNet10/%s/train/%s_%04d.off", classes[i], classes[i], j);
      println("Processing: " + filePath);
      m = new Model(loadStrings(filePath));
      m.readOff();
      
      pg = createGraphics(400, 400, P3D);
      // variable for counting the total number of views that will be saved
      int view = 0;
      for(int theta = 0; theta < numTheta; theta++){
        for(int phi = 1; phi < numPhi; phi++){
          view++;
          // determine the position of the camera using spherical coordinates
          float camPosX = 200 * cos((float) theta/numTheta * TWO_PI) * sin((float) phi/numPhi * PI);
          float camPosY = 200 * sin((float) theta/numTheta * TWO_PI) * sin((float) phi/numPhi * PI);
          float camPosZ = 200 * cos((float) phi/numPhi * PI);
          pg.beginDraw();
          pg.camera(camPosX, camPosY, camPosZ, 0, 0, 0, 0, 1, 0);
          pg.background(0);
          pg.ambientLight(150, 150, 150);
          pg.directionalLight(150, 150, 150, 1, 0, 0);
          pg.directionalLight(150, 150, 150, 0, 1, 0);
          pg.directionalLight(150, 150, 150, 0, 0, 1);
          pg.lightSpecular(210, 210, 210);
          shininess(4.0);
          m.drawModel(pg);
          pg.save(classes[i]+"/"+classes[i]+"_"+j+"/view_"+view+".jpg");
          pg.endDraw();
        }
      }
      int elapsed_time = millis() - currentTime;
      println("Elapsed time: " + elapsed_time);
    }
  }
}

//int numViews = 16;
//void setup(){
//  size(500, 400, P3D);
//  m = new Model(loadStrings("ModelNet10/toilet/train/toilet_0202.off"));
//  m.readOff();
  
//  pg = createGraphics(500, 400, P3D);
//  int currentTime = millis();
//  for(int i = 0; i < numViews; i++){
//    pg.beginDraw();
//    pg.camera(200 * cos((i + 1)*(TWO_PI/numViews)), 0, 200 * sin((i + 1)*(TWO_PI/numViews)), 0, 0, 0, 0, 1, 0);
//    //float camz = (500/2)/tan(PI/6);
//    //pg.perspective(PI/3.0, 500/400, camz/10, camz*10);
//    pg.background(0);
//    pg.lights();
//    m.drawModel(pg);
//    pg.save("view" + i + ".jpg");
//    pg.endDraw();
//  }
//  print(millis() - currentTime);
  
//  // this is for interactive 3D view
//  cam = new PeasyCam(this, 200);
//}

//void draw(){
//  background(0);
//  lights();
//  m.drawModel();
//}

class Model{
  String[] lines;
  PVector[] vertices;
  Polygon[] polys;
  Model(String[] off){
    lines = off;
  }
  // read the OFF file and populate the vertices and polys array
  void readOff(){
    // read the second line in the file to get the number of vertices and faces
    int[] modelInfo = int(split(lines[1], " "));
    vertices = new PVector[modelInfo[0]];
    polys = new Polygon[modelInfo[1]];
    // variable that will store the maximum dist of any vertex from the origin
    float farthestVert = 0;
    // populate the vertices array
    for(int i = 0; i < vertices.length; i++){
      float[] vertInfo = float(split(lines[i + 2], " "));
      vertices[i] = new PVector(vertInfo[0], vertInfo[1], vertInfo[2]);
      // check if the current vertex is farther away from the origin than the known farthest vertex
      if(vertices[i].mag() > farthestVert){
        farthestVert = vertices[i].mag();
      }
    }
    for(int i = 0; i < vertices.length; i++){
      // scale the model to fit inside a unit sphere
      vertices[i].div(farthestVert);
      // scale the model
      vertices[i].mult(100);
    }
    // populate the polygons array
    for(int j = 0; j < polys.length; j++){
      int[] faceInfo = int(split(lines[j + 2 + vertices.length], " "));
      // create a temporary vertices list for each polygon
      // the first number in the faceInfo array is the number of vertices in the face
      PVector[] faceVerts = new PVector[faceInfo[0]];
      for(int k = 0; k < faceInfo.length - 1; k++){
        // the faceInfo array contains the indices to the vertices array for each face
        faceVerts[k] = new PVector(vertices[faceInfo[k + 1]].x, vertices[faceInfo[k + 1]].y, vertices[faceInfo[k + 1]].z);
      }
      polys[j] = new Polygon(faceVerts);
    }
    //printArray(vertices);
    //print(farthestVert);
  }
  void drawModel(){
    for(Polygon p: polys){
      p.drawPoly();
    }
  }
  // method for drawing the model onto a PGraphics object
  void drawModel(PGraphics pg){
    for(Polygon p: polys){
      p.drawPoly(pg);
    }
  }
}

class Polygon{
  PVector[] vertices;
  Polygon(PVector[] v){
    vertices = v;
  }
  void drawPoly(){
    //stroke(100);
    noStroke();
    fill(200);
    beginShape();
    for(PVector v: vertices){
      vertex(v.x, v.y, v.z);
    }
    endShape(CLOSE);
  }
  // method for drawing the polygon onto a PGraphics object
  void drawPoly(PGraphics pg){
    pg.noStroke();
    pg.fill(200);
    pg.beginShape();
    for(PVector v: vertices){
      pg.vertex(v.x, v.y, v.z);
    }
    pg.endShape(CLOSE);
  }
}
