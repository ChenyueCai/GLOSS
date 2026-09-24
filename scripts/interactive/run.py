import argparse
import logging
import os


import gloss.logging

from gloss_interactive.paths import INTERACTIVE_MESH_CONFIG
from gloss_interactive.server import create_server
from tornado.ioloop import IOLoop

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Sample script for rendering a mesh from a fixed viewpoint.')
    parser.add_argument('--port', default=8080, type=int)
    parser.add_argument('--session_name', default="test-session", type=str)
    parser.add_argument('--fill_fov', default=0.4, type=float,
                        help='FOV (radians) used when presampling fill cameras for new meshes.')
    parser.add_argument('--fill_dist', default=0.75, type=float,
                        help='Camera distance used when presampling fill cameras for new meshes.')
    parser.add_argument('--brush_fov', default=0.4, type=float,
                        help='Default FOV (radians) for the (single) brush camera if --brush_fov_dist is not given.')
    parser.add_argument('--brush_dist', default=0.75, type=float,
                        help='Default camera distance for the (single) brush camera if --brush_fov_dist is not given.')
    parser.add_argument('--brush_fov_dist', action='append', default=None,
                        help=('Brush camera (fov, dist) entry. Repeatable: each --brush_fov_dist appends one '
                              '(fov, dist) pair to the stack. Format: "fov,dist", e.g. --brush_fov_dist 0.4,0.75 '
                              '--brush_fov_dist 0.3,1.0. When omitted, the stack is [(brush_fov, brush_dist)].'))
    gloss.logging.add_log_level_flag(parser)
    args = parser.parse_args()

    brush_fov_dist_stack = None
    if args.brush_fov_dist:
        brush_fov_dist_stack = []
        for entry in args.brush_fov_dist:
            try:
                fov_str, dist_str = entry.split(',')
                brush_fov_dist_stack.append((float(fov_str), float(dist_str)))
            except (ValueError, TypeError):
                parser.error(f"--brush_fov_dist expects 'fov,dist' (got '{entry}')")

    gloss.logging.default_log_setup(args.log_level)
    logger.info('Server is starting...')
    config_path = INTERACTIVE_MESH_CONFIG
    server = create_server(config_fp=config_path,
                           session_name=args.session_name,
                           debug_dir=None,
                           fill_fov=args.fill_fov,
                           fill_dist=args.fill_dist,
                           brush_fov=args.brush_fov,
                           brush_dist=args.brush_dist,
                           brush_fov_dist_stack=brush_fov_dist_stack)
    server.listen(args.port)
    IOLoop.instance().start()
