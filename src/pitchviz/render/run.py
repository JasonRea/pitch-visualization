import sys
import traceback

from pitchviz.data.fetch import daily_pitches
from pitchviz.data.filters import (
    pitches_filter,
    pitches_filter_vs_left,
    pitches_filter_vs_right,
    pitches_filter_by_pitch_type,
)
from pitchviz.render.assets import player_bio
from pitchviz.render.builder import VizualizationBuilder

def print_usage():
    print("Usage: python -m pitchviz.render.run [option] [argument]\n")
    print("Options:")
    print("  -d <date> <pitcher> [quality]                   Render all pitches from an outing on the given day")
    print('                                                  Example: python -m pitchviz.render.run -d "2026-02-24" "Ranger Suarez" "high_quality"')
    print("  -dp <date> <pitcher> <pitch_type> [quality]     Render a single pitch type from an outing")
    print('                                                  Example: python -m pitchviz.render.run -dp "2026-02-24" "Ranger Suarez" "FF" "high_quality"')
    print("  -dA <date> [quality]                            Render all splits (all/vs-left/vs-right + per pitch type) for every pitcher on the given day")
    print('                                                  Example: python -m pitchviz.render.run -dA "2026-02-24" "low_quality"')
    sys.exit(1)

if __name__ == '__main__':

    flag  = sys.argv[1] if len(sys.argv) > 1 else None
    arg_1 = sys.argv[2] if len(sys.argv) > 2 else None
    arg_2 = sys.argv[3] if len(sys.argv) > 3 else None
    arg_3 = sys.argv[4] if len(sys.argv) > 4 else None
    arg_4 = sys.argv[5] if len(sys.argv) > 5 else None

    builder = VizualizationBuilder()

    try:
        match flag:
            case "-d":
                scene_class = (
                    builder
                    .load_pitches(date=arg_1, pitcher=arg_2, filter=pitches_filter)
                    .buildm_pitches()
                )
                VizualizationBuilder.render(scene_class, quality=arg_3, filename=f"{arg_2} {arg_1}")

            case "-dl":
                scene_class = (
                    builder
                    .load_pitches(date=arg_1, pitcher=arg_2, filter=pitches_filter_vs_left)
                    .buildm_pitches()
                )
                VizualizationBuilder.render(scene_class, quality=arg_3, filename=f"{arg_2} {arg_1} vs Left")

            case "-dr":
                scene_class = (
                    builder
                    .load_pitches(date=arg_1, pitcher=arg_2, filter=pitches_filter_vs_right)
                    .buildm_pitches()
                )
                VizualizationBuilder.render(scene_class, quality=arg_3, filename=f"{arg_2} {arg_1} vs Right")

            case "-dp":
                scene_class = (
                    builder
                    .load_pitches(date=arg_1, pitcher=arg_2, filter=pitches_filter_by_pitch_type(arg_3))
                    .buildm_pitches()
                )
                VizualizationBuilder.render(scene_class, quality=arg_4, filename=f"{arg_2} {arg_1} {arg_3}")

            case "-da":
                scene_class = (
                    builder
                    .load_pitches(date=arg_1, pitcher=arg_2, filter=pitches_filter)
                    .buildm_pitches()
                )
                VizualizationBuilder.render(scene_class, quality=arg_3, filename=f"{arg_2} {arg_1}")

                scene_class_left = (
                    builder
                    .load_pitches(date=arg_1, pitcher=arg_2, filter=pitches_filter_vs_left)
                    .buildm_pitches()
                )
                VizualizationBuilder.render(scene_class_left, quality=arg_3, filename=f"{arg_2} {arg_1} vs Left")

                scene_class_right = (
                    builder
                    .load_pitches(date=arg_1, pitcher=arg_2, filter=pitches_filter_vs_right)
                    .buildm_pitches()
                )
                VizualizationBuilder.render(scene_class_right, quality=arg_3, filename=f"{arg_2} {arg_1} vs Right")

            case "-dA":
                day_df = daily_pitches(arg_1)
                for pitcher_id, pitcher_df in day_df.groupby("pitcher"):
                    pitcher_name = player_bio(pitcher_id)[0]
                    for filt, label in [
                        (pitches_filter,           ""),
                        (pitches_filter_vs_left,   " vs Left"),
                        (pitches_filter_vs_right,  " vs Right"),
                    ]:
                        builder.load_pitches_from_df(pitcher_df, filt)
                        if builder._axes is None:
                            continue
                        scene_class = builder.buildm_pitches()
                        VizualizationBuilder.render(
                            scene_class,
                            quality=arg_2,
                            filename=f"{pitcher_name} {arg_1}{label}",
                        )

                    pitch_types = pitcher_df["pitch_type"].dropna().unique()
                    for code in pitch_types:
                        filt = pitches_filter_by_pitch_type(code)
                        builder.load_pitches_from_df(pitcher_df, filt)
                        if builder._axes is None:
                            continue
                        scene_class = builder.buildm_pitches()
                        VizualizationBuilder.render(
                            scene_class,
                            quality=arg_2,
                            filename=f"{pitcher_name} {arg_1} {code}",
                        )

            case _:
                print_usage()

    except Exception as e:
        traceback.print_exc()
        print_usage()
