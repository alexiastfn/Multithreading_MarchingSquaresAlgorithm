// Author: APD team, except where source was noted

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <time.h>
#include <unistd.h>

#include "helpers.h"

#define CONTOUR_CONFIG_COUNT 16
#define FILENAME_MAX_SIZE 50
#define STEP 8
#define SIGMA 200
#define RESCALE_X 2048
#define RESCALE_Y 2048

#define CLAMP(v, min, max) \
  if (v < min) {           \
    v = min;               \
  } else if (v > max) {    \
    v = max;               \
  }

struct my_struct {
  // rescale:
  pthread_barrier_t *my_barrier;
  long id;
  int number_threads;
  ppm_image *initial_image;
  ppm_image *final_image;

  // grid:
  int step_x;
  int step_y;
  unsigned char **grid;

  // march:
  ppm_image **contour_map;
};

ppm_image **init_contour_map() {
  ppm_image **map =
      (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
  if (!map) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
    char filename[FILENAME_MAX_SIZE];
    sprintf(filename, "./contours/%d.ppm", i);
    map[i] = read_ppm(filename);
  }

  return map;
}

void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
  for (int i = 0; i < contour->x; i++) {
    for (int j = 0; j < contour->y; j++) {
      int contour_pixel_index = contour->x * i + j;
      int image_pixel_index = (x + i) * image->y + y + j;

      image->data[image_pixel_index].red =
          contour->data[contour_pixel_index].red;
      image->data[image_pixel_index].green =
          contour->data[contour_pixel_index].green;
      image->data[image_pixel_index].blue =
          contour->data[contour_pixel_index].blue;
    }
  }
}

void free_resources(ppm_image *image, ppm_image **contour_map,
                    unsigned char **grid, int step_x) {
  for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
    free(contour_map[i]->data);
    free(contour_map[i]);
  }
  free(contour_map);

  for (int i = 0; i <= image->x / step_x; i++) {
    free(grid[i]);
  }
  free(grid);

  free(image->data);
  free(image);
}

void *my_thread_function(void *arg) {
  struct my_struct *data = (struct my_struct *)arg;
  pthread_barrier_t *my_barrier = data->my_barrier;

  // RESCALE

  ppm_image *image = data->initial_image;
  ppm_image *new_image = data->final_image;
  int ID = data->id;
  int P = data->number_threads;
  int start, end, N;
  uint8_t sample[3];

  if (image->x > RESCALE_X && image->y > RESCALE_Y) {
    N = RESCALE_X;
    start = ID * (double)N / P;
    end = MIN((ID + 1) * (double)N / P, N);

    for (int i = start; i < end; i++) {
      for (int j = 0; j < new_image->y; j++) {
        float u = (float)i / (float)(new_image->x - 1);
        float v = (float)j / (float)(new_image->y - 1);
        sample_bicubic(image, u, v, sample);

        new_image->data[i * new_image->y + j].red = sample[0];
        new_image->data[i * new_image->y + j].green = sample[1];
        new_image->data[i * new_image->y + j].blue = sample[2];
      }
    }

  } else {
    new_image = data->initial_image;
    data->final_image = data->initial_image;
  }

  int result = pthread_barrier_wait(my_barrier);
  if (result != PTHREAD_BARRIER_SERIAL_THREAD && result != 0) {
    printf("Error pthread_barrier_wait() \n");
    pthread_exit(NULL);
  }

  // GRID

  int p = new_image->x / STEP;
  int q = new_image->y / STEP;
  int step_x = data->step_x;
  int step_y = data->step_y;
  unsigned char **grid = data->grid;
  int sigma = SIGMA;
  N = p;

  start = ID * (double)N / P;
  end = MIN((ID + 1) * (double)N / P, N);

  for (int i = start; i < end; i++) {
    for (int j = 0; j < q; j++) {
      ppm_pixel curr_pixel =
          new_image->data[i * step_x * new_image->y + j * step_y];

      unsigned char curr_color =
          (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

      if (curr_color > sigma) {
        grid[i][j] = 0;
      } else {
        grid[i][j] = 1;
      }
    }
  }

  for (int i = start; i < end; i++) {
    ppm_pixel curr_pixel =
        new_image->data[i * step_x * new_image->y + new_image->x - 1];

    unsigned char curr_color =
        (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

    if (curr_color > sigma) {
      grid[i][q] = 0;
    } else {
      grid[i][q] = 1;
    }
  }

  N = q;
  start = ID * (double)N / P;
  end = MIN((ID + 1) * (double)N / P, N);

  for (int j = start; j < end; j++) {
    ppm_pixel curr_pixel =
        new_image->data[(new_image->x - 1) * new_image->y + j * step_y];

    unsigned char curr_color =
        (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

    if (curr_color > sigma) {
      grid[p][j] = 0;
    } else {
      grid[p][j] = 1;
    }
  }

  result = pthread_barrier_wait(my_barrier);
  if (result != PTHREAD_BARRIER_SERIAL_THREAD && result != 0) {
    printf("Error pthread_barrier_wait() \n");
    pthread_exit(NULL);
  }

  // MARCH

  N = p;
  ppm_image **contour_map = data->contour_map;
  start = ID * (double)N / P;
  end = MIN((ID + 1) * (double)N / P, N);

  for (int i = start; i < end; i++) {
    for (int j = 0; j < q; j++) {
      unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] +
                        2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
      update_image(new_image, contour_map[k], i * step_x, j * step_y);
    }
  }

  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
    return 1;
  }

  int step_x = STEP;
  int step_y = STEP;

  ppm_image *image = read_ppm(argv[1]);
  ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
  if (!new_image) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  new_image->x = RESCALE_X;
  new_image->y = RESCALE_Y;
  new_image->data =
      (ppm_pixel *)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
  if (!new_image->data) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  int p = image->x / step_x;
  int q = image->y / step_y;

  unsigned char **grid =
      (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
  if (!grid) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  for (int i = 0; i <= p; i++) {
    grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
    if (!grid[i]) {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(1);
    }
  }

  int num_threads = atoi(argv[3]);
  pthread_t threads[num_threads];

  struct my_struct structs[num_threads];
  unsigned count = num_threads;

  pthread_barrier_t my_barrier;
  int result = pthread_barrier_init(&my_barrier, NULL, count);
  if (result) {
    printf("Error barrier_init() \n");
    exit(-1);
  }
  ppm_image **contour_map = init_contour_map();

  for (int i = 0; i < num_threads; i++) {
    structs[i].id = i;
    structs[i].number_threads = num_threads;
    structs[i].final_image = new_image;
    structs[i].initial_image = image;
    structs[i].grid = grid;
    structs[i].step_x = step_x;
    structs[i].step_y = step_y;
    structs[i].contour_map = contour_map;
    structs[i].my_barrier = &my_barrier;

    result = pthread_create(&threads[i], NULL, my_thread_function, &structs[i]);

    if (result) {
      printf("Error pthread_create() \n");
      exit(-1);
    }
  }

  void *status;
  for (int id = 0; id < num_threads; id++) {
    result = pthread_join(threads[id], &status);

    if (result) {
      printf("Error pthread_join() for the %d-th thread \n", id);
      exit(-1);
    }
  }

  result = pthread_barrier_destroy(&my_barrier);
  if (result) {
    printf("Error pthread_barrier_destroy() \n");
    exit(-1);
  }

  write_ppm(structs[0].final_image, argv[2]);
  free_resources(new_image, contour_map, grid, step_x);

  return 0;
}
