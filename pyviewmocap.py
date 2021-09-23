import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from typing import Dict
import numpy as np
import pandas as pd

def dataframe_to_dict_of_arrays(dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
	From kinetics toolkit - Author: 

    Convert a pandas DataFrame to a dict of numpy ndarrays.

    This function mirrors the dict_of_arrays_to_dataframe function. It is
    mainly used by the TimeSeries.from_dataframe method.

    Parameters
    ----------
    pd_dataframe
        The dataframe to be converted.

    Returns
    -------
    Dict[str, np.ndarray]


    Examples
    --------
    In the simplest case, each dataframe column becomes a dict key.

        >>> df = pd.DataFrame([[0, 3], [1, 4], [2, 5]])
        >>> df.columns = ['column1', 'column2']
        >>> df
           column1  column2
        0        0        3
        1        1        4
        2        2        5

        >>> data = dataframe_to_dict_of_arrays(df)

        >>> data['column1']
        array([0, 1, 2])

        >>> data['column2']
        array([3, 4, 5])

    If the dataframe contains similar column names with indices in brackets
    (for example, Forces[0], Forces[1], Forces[2]), then these columns are
    combined in a single array.

        >>> df = pd.DataFrame([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
        >>> df.columns = ['Forces[0]', 'Forces[1]', 'Forces[2]', 'Other']
        >>> df
           Forces[0]  Forces[1]  Forces[2]  Other
        0          0          3          6      9
        1          1          4          7     10
        2          2          5          8     11

        >>> data = dataframe_to_dict_of_arrays(df)

        >>> data['Forces']
        array([[0, 3, 6],
               [1, 4, 7],
               [2, 5, 8]])

        >>> data['Other']
        array([ 9, 10, 11])

    """
    # Remove spaces in indexes between brackets
    columns = dataframe.columns
    new_columns = []
    for i_column, column in enumerate(columns):
        splitted = column.split('[')
        if len(splitted) > 1:  # There are brackets
            new_columns.append(
                splitted[0] + '[' + splitted[1].replace(' ', ''))
        else:
            new_columns.append(column)
    dataframe.columns = columns

    # Search for the column names and their dimensions
    # At the end, we end with something like:
    #    dimensions['Data1'] = []
    #    dimensions['Data2'] = [[0], [1], [2]]
    #    dimensions['Data3'] = [[0,0],[0,1],[1,0],[1,1]]
    dimensions = dict()  # type: Dict[str, List]
    for column in dataframe.columns:
        splitted = column.split('[')
        if len(splitted) == 1:  # No brackets
            dimensions[column] = []
        else:  # With brackets
            key = splitted[0]
            index = literal_eval('[' + splitted[1])

            if key in dimensions:
                dimensions[key].append(index)
            else:
                dimensions[key] = [index]

    n_samples = len(dataframe)

    # Assign the columns to the output
    out = dict()  # type: Dict[str, np.ndarray]
    for key in dimensions:
        if len(dimensions[key]) == 0:
            out[key] = dataframe[key].to_numpy()
        else:
            highest_dims = np.max(np.array(dimensions[key]), axis=0)

            columns = [key + str(dim).replace(' ', '')
                       for dim in sorted(dimensions[key])]
            out[key] = dataframe[columns].to_numpy()
            out[key] = np.reshape(out[key],
                                  [n_samples] + (highest_dims + 1).tolist())

    return out

def view(df, edges=None, skip=20, save=None, fps=60):
	"""
	View df
	
	Creates the 3D figure and animates it with the input data.
	Args:
		data (list): List of the data positions at each iteration.
		save (bool): Whether to save the recording of the animation. (Default to False).

	From stackoverflow
	"""
	data = ktk.timeseries.dataframe_to_dict_of_arrays(df)

	# Attaching 3D axis to the figure
	fig = plt.figure()
	ax = p3.Axes3D(fig, auto_add_to_figure=False)
	fig.add_axes(ax)

	# if skip: # skip every few frames to view faster
	first_index = list(data.keys())[0]
	length = len(data[first_index])
	indices = np.arange(0, length, skip, dtype=np.int32)
	data = {key: data[key][indices] for key in data.keys()}

	# Initialize scatters
	size=20
	scatters = []
	for key in data.keys():
		if key in ['COM', 'COP', 'XCOM']: size = 300
		else: size = 20
		scatters.append(
			ax.scatter(data[key][0][0:1], data[key][0][1:2], data[key][0][2:3], s=size)
		)
		
	anim_args = [data, scatters, edges]
	
	if edges:
		segments = [(data[start][0][0:3], data[end][0][0:3]) for start, end in edges]
		lines = [ax.plot([start[0], end[0]], [start[1], end[1]], zs=[start[2], end[2]], lw=5) for start, end in segments]
		anim_args.append(lines)

	# Number of iterations
	iterations = len(data[list(data.keys())[0]])

	# Setting the axes properties
	ax.set_xlim3d([-2000, 2000])
	ax.set_xlabel('X')

	ax.set_ylim3d([-3000, 3000])
	ax.set_ylabel('Y')

	ax.set_zlim3d([0, 2000])
	ax.set_zlabel('Z')

	ax.set_title('3D Animated Scatter Example')

	# Provide starting angle for the view.
	ax.view_init(20, 45)

	
	ani = animation.FuncAnimation(fig, _update_points, iterations, fargs=anim_args,
									interval=50, blit=False, repeat=True)

	if save:
		print('SAVING...')
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=900, extra_args=['-vcodec', 'libx264'])
		ani.save(f'{save}.mp4', writer=writer)
		print(f'Saved: {save}.mp4')
	else:
		plt.show()

def _update_points(iteration, data, scatters, edges, lines=None):
	"""
	Update the data held by the scatter plot and therefore animates it.
	Args:
		iteration (int): Current iteration of the animation
		data (list): List of the data positions at each iteration.
		scatters (list): List of all the scatters (One per element)
	Returns:
		list: List of scatters (One per element) with new coordinates
	"""
	for i, k in enumerate(data.keys()):
		scatters[i]._offsets3d = data[k][iteration][0:1], data[k][iteration][1:2], data[k][iteration][2:3]
	
	if edges:
		segments = [(data[start][iteration][0:3], data[end][iteration][0:3]) for start, end in edges]

		for i, (start, end) in enumerate(segments):
			line = lines[i][0]

			line.set_data(np.array([start[0], end[0]]), np.array([start[1], end[1]]))
			line.set_3d_properties(np.array([start[2], end[2]]))
