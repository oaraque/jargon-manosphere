import os
import sys
import argparse
import logging
import json
from pprint import pprint

import neologism_discovery

def main():
    parser = argparse.ArgumentParser(description='Run the neologism discovery method')
    parser.add_argument(
        '--level',
        metavar='logging_level',
        type=str,
        default='INFO',
        help='logging level',
    )
    parser.add_argument(
        '--custom-config',
        '-c',
        metavar='custom_config',
        type=str,
        default=None,
        help='custom config from path',
    )
    parser.add_argument(
        '--input',
        '-i',
        action='store_true',
        default=False,
        help='config from stdin',
    )
    parser.add_argument(
        '--no-emb-filtering',
        action='store_false',
        default=True,
        help='not embedding filtering',
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        default=False,
        help='show LDA plots',
    )
    parser.add_argument(
        '--override',
        action='store_true',
        default=False,
        help='do override of candidates'
    )
    parser.add_argument(
        '--override-method',
        '-o',
        metavar='override_method',
        type=str,
        default='random',
        help='override method: random of seed'
    )
    parser.add_argument(
        '--override-seeds',
        metavar='override_seeds',
        type=str,
        default=None,
        help='path of override seeds',
    )
    parser.add_argument(
        '--override-size',
        metavar='override_size',
        type=int,
        default=100,
        help='default override size',
    )
    parser.add_argument(
        '--no-emb-plot',
        action='store_false',
        default=True,
        help='draw embedding plot',
    )
    parser.add_argument(
        '--no-save',
        action='store_false',
        default=True,
        help='save the resulting candidates',
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.level))
    logger = logging.getLogger(__name__)

    logger.debug('Arguments: {}'.format(vars(args)))

    if args.input:
        # input config overrides file config
        args.custom_config = None 
        config = json.load(sys.stdin)
        logger.info('config loaded from stdin')

    elif not args.custom_config is None:
        if not os.path.exists(args.custom_config):
            raise ValueError
        with open(args.custom_config, 'r') as f:
            config = json.load(f)
        logger.info('config loaded from file {}'.format(args.custom_config))

    else:
        config=None
        logger.info('using default config')

    logger.debug('config: {}'.format(config))

    # override seeds
    if args.override:
        if not args.override_seeds is None:
            if not os.path.exists(args.override_seeds):
                raise ValueError
            with open(args.override_seeds, 'r') as f:
                seeds = f.readlines()
        else:
            seeds = None
    else:
        seeds = None

    _, report = neologism_discovery.run(
        custom_config=config,
        do_embedding_filtering=args.no_emb_filtering,
        show_plots=args.show_plots,
        override=args.override,
        override_method=args.override_method,
        override_size=args.override_size,
        override_seeds=seeds,
        embedding_plot=args.no_emb_plot,
        do_save=args.no_save,
    )

    pprint(report)

if __name__ == '__main__':
    main()