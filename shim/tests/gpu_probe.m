// Measure how long the GPU takes to run a trivial command buffer, forever.
//
//   gpu_probe [interval_seconds]
//
// Prints one latency in milliseconds per line. guard.py runs it next to a test job: when a
// job monopolizes the GPU, the probe's tiny blit waits behind it, the lines stop, and the
// guard kills the job before WindowServer (which waits on the same GPU) trips its watchdog.
// Exits when its parent goes away.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
  @autoreleasepool {
    double interval = argc > 1 ? atof(argv[1]) : 0.5;
    pid_t parent = getppid();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      fprintf(stderr, "gpu_probe: no Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> buffer = [device newBufferWithLength:256 options:MTLResourceStorageModePrivate];
    setvbuf(stdout, NULL, _IOLBF, 0);
    while (getppid() == parent) {
      @autoreleasepool {
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        [blit fillBuffer:buffer range:NSMakeRange(0, 256) value:1];  // forces real GPU work
        [blit endEncoding];
        NSDate* start = [NSDate date];
        [commands commit];
        [commands waitUntilCompleted];
        if (printf("%.1f\n", -[start timeIntervalSinceNow] * 1e3) < 0) {
          return 0;  // stdout closed
        }
      }
      usleep((useconds_t)(interval * 1e6));
    }
  }
  return 0;
}
